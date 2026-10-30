#-- import modules --#
import sys
import argparse
import pysam
import re
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from collections import defaultdict

#-- constant variables --#
CODON_TABLE = { 'TTT':'F', 'TCT':'S', 'TAT':'Y', 'TGT':'C',
                'TTC':'F', 'TCC':'S', 'TAC':'Y', 'TGC':'C',
                'TTA':'L', 'TCA':'S', 'TAA':'*', 'TGA':'*',
                'TTG':'L', 'TCG':'S', 'TAG':'*', 'TGG':'W',
                'CTT':'L', 'CCT':'P', 'CAT':'H', 'CGT':'R',
                'CTC':'L', 'CCC':'P', 'CAC':'H', 'CGC':'R',
                'CTA':'L', 'CCA':'P', 'CAA':'Q', 'CGA':'R',
                'CTG':'L', 'CCG':'P', 'CAG':'Q', 'CGG':'R',
                'ATT':'I', 'ACT':'T', 'AAT':'N', 'AGT':'S',
                'ATC':'I', 'ACC':'T', 'AAC':'N', 'AGC':'S',
                'ATA':'I', 'ACA':'T', 'AAA':'K', 'AGA':'R',
                'ATG':'M', 'ACG':'T', 'AAG':'K', 'AGG':'R',
                'GTT':'V', 'GCT':'A', 'GAT':'D', 'GGT':'G',
                'GTC':'V', 'GCC':'A', 'GAC':'D', 'GGC':'G',
                'GTA':'V', 'GCA':'A', 'GAA':'E', 'GGA':'G',
                'GTG':'V', 'GCG':'A', 'GAG':'E', 'GGG':'G' }

#-- functions --#
def extract_read_info(read: pysam.AlignedSegment) -> dict:
    """
    Parse pysam read object to extract relevant information.
    Parameters:
        -- read: a pysam AlignedSegment
    Returns:
        -- dict: a read dict
    """
    if read.is_reverse:
        read_seq = str(Seq(read.query_sequence).reverse_complement())
        read_qual = read.query_qualities[::-1]
    else:
        read_seq = read.query_sequence
        read_qual = read.query_qualities
    
    return {
        'ref':   read.reference_name,
        'pos':   read.reference_start,
        'seq':   read_seq,
        'qual':  read_qual, 
        'md':    read.get_tag('MD'),
        'cigar': read.cigartuples
    }

def parse_md(read: dict) -> list:
    """
    Parse a read to extract variants and codon information within the ORF range.
    Parameters:
        -- read: a read dict
    Returns: 
        -- list: list of variant tuples (type, position, variant)
    """
    read_pos = read.get('pos')
    read_md = read.get('md')
    md_pattern = re.finditer(r'(\d+)|([A-Z]+)|(\^[A-Z]+)', read_md)

    # note read_pos is 0 base
    base_ref_pos = read_pos
    
    variants = []
    for match in md_pattern:
        if match.group(1):
            num_matches = int(match.group(1))
            base_ref_pos += num_matches
        elif match.group(2):
            variants.append(('X', base_ref_pos, match.group(2)))
            base_ref_pos += 1
        elif match.group(3):
            deleted_bases = match.group(3)[1:]
            for base in deleted_bases:
                variants.append(('D', base_ref_pos, base))
                base_ref_pos += 1

    return variants

def group_variants(variants: list) -> list:
    """
    Group variants by type, reference name, codon index, and codon.
    Parameters:
        -- variants: list of variant dicts
    Returns:
        -- list: list of grouped variant dicts
    """
    variants_grouped = defaultdict(list)
    for var in variants:
        key = (var['var_type'], var['ref_name'], var['codon_idx'], var['ref_codon'])
        variants_grouped[key].append(var)

    variants_merged = []
    for (var_type, ref_name, codon_idx, ref_codon), group in variants_grouped.items():
        group = sorted(group, key=lambda v: v['ref_pos'])

        ref_pos = group[0]['ref_pos']
        ref_bases = ''.join(v['ref_base'] for v in group)
        alt_bases = ''.join(v['alt_base'] for v in group)
        alt_quals = ','.join(str(v['alt_qual']) for v in group)
        alt_idx = ''.join(str(v['alt_idx']) for v in group)
        base_slots = ['-'] * 3
        qual_slots = ['-'] * 3
        alt_codon = list(ref_codon)
        for v in group:
            base_pos = int(v['alt_idx']) - 1  # Convert 1-based to 0-based
            base_slots[base_pos] = v['alt_base']
            qual_slots[base_pos] = v['alt_qual']
            alt_codon[base_pos] = v['alt_base']

        variants_merged.append({ 'var_type':   var_type,
                                 'ref_name':   ref_name,
                                 'ref_pos':    ref_pos,
                                 'ref_base':   ref_bases,
                                 'alt_base':   alt_bases,
                                 'alt_qual':   alt_quals,
                                 'alt_idx':    alt_idx,
                                 'codon_idx':  codon_idx,
                                 'ref_codon':  ref_codon,
                                 'alt_codon':  ''.join(alt_codon),
                                 'alt_base_1': base_slots[0],
                                 'alt_base_2': base_slots[1],
                                 'alt_base_3': base_slots[2],
                                 'alt_qual_1': qual_slots[0],
                                 'alt_qual_2': qual_slots[1],
                                 'alt_qual_3': qual_slots[2] })
    return variants_merged

def parse_read(read: dict, orf_start: int, orf_end: int, base_qual: int) -> list:
    """
    Parse a read to extract variants and codon information within the ORF range.
    Parameters:
        -- read: a read dict
        -- orf_start: start position of the ORF (1-based)
        -- orf_end: end position of the ORF (1-based)
        -- base_qual: minimum base quality to consider a variant
    Returns: 
        -- list: list of variant tuples
    """
    read_ref = read.get('ref')
    read_pos = read.get('pos')
    read_seq = read.get('seq')
    read_qual = read.get('qual')
    read_cigar = read.get('cigar')
    read_md_variants = parse_md(read)

    if not read_md_variants:
        return []

    base_ref_pos = read_pos
    base_read_pos = 0
    md_index = 0
    variants = []
    for op, op_len in read_cigar:
        if op == 0: # M, match
            for i in range(op_len): 
                if md_index < len(read_md_variants) and read_md_variants[md_index][1] == base_ref_pos:
                    var_type, _, ref_base = read_md_variants[md_index]
                    alt_base = read_seq[base_read_pos]
                    alt_qual = read_qual[base_read_pos]
                    alt_idx = (base_ref_pos - orf_start) % 3 + 1
                    codon_idx = ((base_ref_pos - orf_start) // 3) + 1
                    variants.append({ 'var_type':  var_type,
                                      'ref_name':  read_ref,
                                      'ref_pos':   base_ref_pos,
                                      'ref_base':  ref_base,
                                      'alt_base':  alt_base,
                                      'alt_qual':  alt_qual,
                                      'alt_idx' :  alt_idx,
                                      'codon_idx': codon_idx })
                    md_index += 1
                base_ref_pos += 1
                base_read_pos += 1
        elif op == 1: # I, insertion
            # skip for now, due to complexity of handling alt_qual and codon index
            # alt_base could be longer than 1 base, and codon_index doesn't apply
            # ins_base = read_seq[base_read_pos:base_read_pos + op_len]
            # ins_qual = read_qual[base_read_pos:base_read_pos + op_len]
            # variants.append({ 'var_type':  'I',
            #                   'ref_name':  read_ref,
            #                   'ref_pos':   base_ref_pos,
            #                   'ref_base':  '-',
            #                   'alt_base':  ins_base,
            #                   'alt_qual':  99,
            #                   'codon_idx': '-' })
            base_read_pos += op_len
        elif op == 2: # D, deletion
            for i in range(op_len):
                if md_index < len(read_md_variants) and read_md_variants[md_index][1] == base_ref_pos:
                    var_type, _, ref_base = read_md_variants[md_index]
                    alt_idx = (base_ref_pos - orf_start) % 3 + 1
                    codon_idx = ((base_ref_pos - orf_start) // 3) + 1
                    variants.append({ 'var_type':  'D',
                                      'ref_name':  read_ref,
                                      'ref_pos':   base_ref_pos,
                                      'ref_base':  ref_base,
                                      'alt_base':  '-',
                                      'alt_qual':  99,
                                      'alt_idx' :  alt_idx,
                                      'codon_idx': codon_idx })
                    md_index += 1
                base_ref_pos += 1
        elif op == 3: # N, skip
            base_ref_pos += op_len
        elif op == 4: # S, softclip
            base_read_pos += op_len
        elif op == 5: # H, hardclip
            continue

    if not variants:
        return []

    variants_filtered = []
    for var in variants:
        if var['ref_pos'] >= orf_start and var['ref_pos'] <= orf_end:
            if var['alt_qual'] >= base_qual:
                ref_codon = codon_dict[var['ref_name']][var['codon_idx']]
                variants_filtered.append({ 'var_type':  var['var_type'],
                                           'ref_name':  var['ref_name'],
                                           'ref_pos':   var['ref_pos'],
                                           'ref_base':  var['ref_base'],
                                           'alt_base':  var['alt_base'],
                                           'alt_qual':  var['alt_qual'],
                                           'alt_idx' :  var['alt_idx'],
                                           'codon_idx': var['codon_idx'],
                                           'ref_codon': ref_codon })
    return group_variants(variants_filtered)

def batch_parse_reads(batch_reads: list, orf_start: int, orf_end: int, base_qual: int):
    """
    Process a batch of read pairs to extract variants and barcodes.
    Parameters:
        -- batch_reads: list of read dicts
        -- orf_start: start position of the ORF (1-based)
        -- orf_end: end position of the ORF (1-based)
        -- base_qual: minimum base quality to consider a variant
    Returns:
        -- list: list of variant dicts for the batch
    """
    results = []
    for read in batch_reads:
        result = parse_read(read, orf_start, orf_end, base_qual)
        results.append(result)
    return results

def function_for_processpool(args):
    """
    Wrapper function for process pool as ProcessPoolExecutor expects a function rather than returned results.
    """
    return batch_parse_reads(*args)

def read_bam_in_chunk(bam_path: str, orf_range: str, base_qual: int, chunk_size: int, threads: int):
    """
    Read BAM file in chunks and extract base qualities for reads within the specified ORF range.
    Each chunk is processed in batches using multiprocessing.
    Parameters:
        -- bam_path: path to the BAM file
        -- orf_range: ORF range in the format 'start-end' (1-based)
        -- base_qual: minimum base quality to consider a variant
        -- chunk_size: number of reads to process in each chunk
        -- threads: number of threads for multiprocessing
    Yields:
        -- list: list of variant dicts for each chunk
    """
    orf_start, orf_end = map(int, orf_range.split('-'))

    with ProcessPoolExecutor(max_workers=threads) as executor, pysam.AlignmentFile(bam_path, "rb", threads = threads) as bam_file:
        read_chunk = []
        for read in bam_file.fetch(until_eof=True):
            # skip unampped reads or no MD tag
            if read.is_unmapped or not read.has_tag('MD'):
                continue

            # skip perfectly aligned reads
            if re.fullmatch(r'\d+', read.get_tag('MD')):
                continue

            read_chunk.append(extract_read_info(read))

            if len(read_chunk) >= chunk_size:
                batch_size = min(chunk_size, 5000)
                read_batches = [
                    read_chunk[i:i + batch_size] 
                    for i in range(0, len(read_chunk), batch_size)
                ]

                args_list = [
                    (batch, orf_start, orf_end, base_qual) 
                    for batch in read_batches
                ]

                batch_results = list(executor.map(function_for_processpool, args_list))

                results = [item for batch in batch_results for item in batch]
                yield results

                read_chunk = []

        # Process any remaining reads after file ends
        if read_chunk:
            batch_size = min(chunk_size, 5000)
            read_batches = [
                read_chunk[i:i + batch_size] 
                for i in range(0, len(read_chunk), batch_size)
            ]

            args_list = [
                (batch, orf_start, orf_end, base_qual) 
                for batch in read_batches
            ]

            batch_results = list(executor.map(function_for_processpool, args_list))

            results = [item for batch in batch_results for item in batch]
            yield results

#-- main execution --#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Extract variants and base qualities from a bam file.", allow_abbrev = False)
    parser.add_argument("-i", "--input_bam",  type = str, required = True, help = "Input BAM file")
    parser.add_argument("-r", "--reference",  type = str, required = True, help = "Reference FASTA file")
    parser.add_argument("-o", "--orf_range",  type = str, required = True, help = "ORF range in the reference 0-based (e.g., '352-1383')")
    parser.add_argument("-p", "--prefix",     type = str, required = True, help = "Output prefix")
    parser.add_argument("-b", "--base_qual",  type = int, default = 20,    help = "Minimum base quality")
    parser.add_argument("-c", "--chunk_size", type = int, default = 1000,  help = "Chunk size for processing reads")
    parser.add_argument("-t", "--threads",    type = int, default = 4,     help = "Number of threads")
    
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"Error: Unrecognized arguments: {' '.join(unknown)}", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    ref_dict = SeqIO.to_dict(SeqIO.parse(args.reference, "fasta"))

    orf_dict = {}
    codon_dict = {}
    orf_start, orf_end = map(int, args.orf_range.split('-'))
    for chrom, record in ref_dict.items():
        # need to be careful with 1-based vs 0-based indexing
        # orf_start and orf_end should be 0-based too
        orf_seq = record.seq[orf_start : orf_end + 1]
        orf_dict[chrom] = orf_seq
        ref_codons = {}
        for i in range(0, len(orf_seq) - 2, 3):
            codon_idx = i // 3 + 1 # 1-based index for codon
            ref_codon = orf_seq[i:i + 3]
            if len(ref_codon) == 3:
                ref_codons[codon_idx] = ref_codon
        codon_dict[chrom] = ref_codons

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Extracting read information, please wait...", flush = True)
    variant_results = []
    for i, result_chunk in enumerate(read_bam_in_chunk(args.input_bam, args.orf_range, args.base_qual, args.chunk_size, args.threads)):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |--> Processed chunk {i+1} with {len(result_chunk)} read pairs", flush = True)
        variant_results.extend(result_chunk)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |--> Finished.", flush = True)

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Creating the variant matrix, please wait...", flush = True)
    variant_flattened = [var for var_list in variant_results for var in var_list]
    df_variant_flattened = pd.DataFrame(variant_flattened)

    output_file = f"{args.prefix}_variants.tsv"
    df_variant_flattened.to_csv(output_file, sep = '\t', index = False)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |--> Finished.", flush = True)

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Counting the variants, please wait...", flush = True)
    drop_list = ['alt_qual', 'alt_qual_1', 'alt_qual_2', 'alt_qual_3']
    df_variant_counted = df_variant_flattened.drop(columns=drop_list).value_counts().reset_index(name='count')
    df_variant_counted.sort_values(by=['ref_name', 'ref_pos'], inplace=True)

    counter_file = f"{args.prefix}_counts.tsv"
    df_variant_counted.to_csv(counter_file, sep = '\t', index = False)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |--> Finished.", flush = True)