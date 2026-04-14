#-- import modules --#
import os
import sys
import argparse
import re
import gc
import pysam
import re
import numpy as np
import polars as pl
from Bio import SeqIO
from Bio.Seq import Seq
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from collections import defaultdict

#-- functions --#
def get_base_cov(bam_path: str, chrom: str, start: int, end: int, qual: int) -> dict:
    """
    Compute per-base coverage for a region and return a dict {1-based position: coverage}
    Parameters:
        -- bam_path: path to the BAM file
        -- chrom: chromosome name
        -- start: start position (0-based)
        -- end: end position (0-based)
        -- qual: quality threshold
    Returns:
        -- dict: {1-based position: coverage} dict for ORF region
    """
    bam = pysam.AlignmentFile(bam_path, "rb")
    A, C, G, T = bam.count_coverage(chrom, start, end, quality_threshold = qual)
    bam.close()

    # total coverage per base
    coverage_array = np.array(A, dtype=np.uint32) + \
                     np.array(C, dtype=np.uint32) + \
                     np.array(G, dtype=np.uint32) + \
                     np.array(T, dtype=np.uint32)

    dict_base_cov = {pos + 1 + start: cov for pos, cov in enumerate(coverage_array)}
    return dict_base_cov

def chunk_ranges(start: int, end: int, sub_region_size: int) -> tuple:
    """
    Yield (start, end) tuples for chunking a large region.
    Parameters:
        -- start: start position (0-based)
        -- end: end position (0-based)
        -- sub_region_size: size of each chunk
    Yields:
        -- tuple: (start, end) for each chunk
    """
    for s in range(start, end, sub_region_size):
        yield s, min(s + sub_region_size, end)

def get_base_cov_in_chunk(bam_path: str, chrom: str, start: int, end: int, qual: int, sub_region_size: int, threads: int) -> dict:
    """
    Compute per-base coverage for a large region using multiple processes.
    Parameters:
        -- bam_path: path to the BAM file
        -- chrom: chromosome name
        -- start: start position (0-based)
        -- end: end position (0-based)
        -- qual: quality threshold
        -- sub_region_size: size of each chunk to process in parallel
        -- threads: number of threads for parallel processing
    Returns:
        -- dict: a dict {1-based position: coverage}
    """
    chunks = list(chunk_ranges(start, end, sub_region_size))
    dicts = []

    with ProcessPoolExecutor(max_workers = threads) as executor:
        futures = [executor.submit(get_base_cov, bam_path, chrom, start, end, qual) for start, end in chunks]
        for f in futures:
            dicts.append(f.result())

    merged_dict = {}
    for d in dicts:
        merged_dict.update(d)

    return merged_dict

def init_worker():
    """
    Initilize worker
    """
    global global_dict_base_cov
    global_dict_base_cov = dict_base_cov

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

def gatk_formating(variants: list) -> list:
    """
    Format variants into GATK-like format for easier downstream processing.
    Parameters:
        -- variants: list of variant dicts
    Returns:
        -- list: list of formatted variant dicts
    """
    varying_bases = 0
    varying_codons = 0
    base_mut = ""
    codon_mut = ""
    aa_mut = ""
    pos_mut = ""

    # --- average coverage ---
    base_pos = [ v['ref_pos'] for v in variants if v['var_type'] == 'X' ]
    base_cov = [ global_dict_base_cov.get(pos, 0) for pos in base_pos ]
    base_cov_avg = int(np.mean(base_cov)) if base_cov else 0

    # --- base variants ---
    base_variants = [ v for v in variants if v['var_type'] == 'X' ]
    varying_bases = len(base_variants)

    base_mut = ", ".join(
        f"{v['ref_pos']}:{v['ref_base']}>{v['alt_base']}"
        for v in base_variants
    )

    # --- codon variants ---
    codon_changes = defaultdict(list)
    for v in base_variants:
        codon_idx = ((v['ref_pos'] - orf_start) // 3) + 1
        codon_changes[codon_idx].append(v)    

    varying_codons = len(codon_changes)

    codon_mut_list = []
    aa_mut_list = []
    pos_mut_list = []

    for codon_idx in sorted(codon_changes):
        vars_in_codon = codon_changes[codon_idx]
        ref_codon = str(codon_dict[chrom][codon_idx])
        codon_list = list(ref_codon)

        for v in vars_in_codon:
            pos_in_codon = (v['ref_pos'] - orf_start) % 3
            codon_list[pos_in_codon] = v['alt_base']

        alt_codon = "".join(codon_list)

        # --- codon_mut ---
        codon_mut_list.append(f"{codon_idx}:{ref_codon}>{alt_codon}")

        # --- translate ---
        ref_aa = str(Seq(ref_codon).translate())
        alt_aa = str(Seq(alt_codon).translate())

        # --- mutation type ---
        if alt_aa == ref_aa:
            mut_type = "S"
        elif alt_aa == "*":
            mut_type = "N"
        else:
            mut_type = "M"

        # --- aa_mut ---
        aa_mut_list.append(f"{mut_type}:{ref_aa}>{alt_aa}")

        # --- pos_mut ---
        pos_mut_list.append(f"{ref_aa}{codon_idx}{alt_aa}")

    codon_mut = ", ".join(codon_mut_list)
    aa_mut = ", ".join(aa_mut_list)
    pos_mut = ";".join(pos_mut_list)

    return base_cov_avg, varying_bases, base_mut, varying_codons, codon_mut, aa_mut, pos_mut

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
    
    return gatk_formating(variants_filtered)

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

    with pysam.AlignmentFile(bam_path, "rb", threads = threads) as bam_file, \
        ProcessPoolExecutor(max_workers = threads, initializer = init_worker) as executor:

        batch_size = min(chunk_size, 5000)

        # process reads in chunks
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
                read_batches = [
                    read_chunk[i:i + batch_size] 
                    for i in range(0, len(read_chunk), batch_size)
                ]

                futures = [ 
                    executor.submit(function_for_processpool, (batch, orf_start, orf_end, base_qual)) 
                    for batch in read_batches 
                ]

                results = []
                for f in futures:
                    batch_result = f.result()
                    if batch_result:
                        df_batch = pl.DataFrame(batch_result, schema={
                            "base_cov_avg": pl.Int64,
                            "varying_bases": pl.Utf8,
                            "base_mut": pl.Utf8,
                            "varying_codons": pl.Utf8,
                            "codon_mut": pl.Utf8,
                            "aa_mut": pl.Utf8,
                            "pos_mut": pl.Utf8
                        }, orient = "row")
                        results.append(df_batch)
                    # -- free memory -- #
                    del batch_result
                    gc.collect()

                if results:
                    df_yield = pl.concat(results, how = "vertical", rechunk = True)
                    df_yield = df_yield.with_columns(pl.len().over("base_mut").alias("counts"))
                else:
                    df_yield = pl.DataFrame([], schema={
                        "base_cov_avg": pl.Int64,
                        "varying_bases": pl.Utf8,
                        "base_mut": pl.Utf8,
                        "varying_codons": pl.Utf8,
                        "codon_mut": pl.Utf8,
                        "aa_mut": pl.Utf8,
                        "pos_mut": pl.Utf8,
                        "counts": pl.Int64
                    }, orient = "row")

                read_chunk = []

                # -- free memory -- #
                del read_batches, futures, results
                gc.collect()

                yield df_yield

        # Process any remaining reads after file ends
        if read_chunk:
            read_batches = [
                read_chunk[i:i + batch_size] 
                for i in range(0, len(read_chunk), batch_size)
            ]

            futures = [ 
                executor.submit(function_for_processpool, (batch, orf_start, orf_end, base_qual)) 
                for batch in read_batches 
            ]

            results = []
            for f in futures:
                batch_result = f.result()
                if batch_result:
                    df_batch = pl.DataFrame(batch_result, schema={
                        "base_cov_avg": pl.Int64,
                        "varying_bases": pl.Utf8,
                        "base_mut": pl.Utf8,
                        "varying_codons": pl.Utf8,
                        "codon_mut": pl.Utf8,
                        "aa_mut": pl.Utf8,
                        "pos_mut": pl.Utf8
                    }, orient = "row")
                    results.append(df_batch)
                # -- free memory -- #
                del batch_result
                gc.collect()

            if results:
                df_yield = pl.concat(results, how = "vertical", rechunk = True)
                df_yield = df_yield.with_columns(pl.len().over("base_mut").alias("counts"))
            else:
                df_yield = pl.DataFrame([], schema={
                    "base_cov_avg": pl.Int64,
                    "varying_bases": pl.Utf8,
                    "base_mut": pl.Utf8,
                    "varying_codons": pl.Utf8,
                    "codon_mut": pl.Utf8,
                    "aa_mut": pl.Utf8,
                    "pos_mut": pl.Utf8,
                    "counts": pl.Int64
                }, orient = "row")
            
            # -- free memory -- #
            del read_chunk, read_batches, futures, results
            gc.collect()           

            yield df_yield
                
#-- main execution --#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Extract variants and base qualities from a bam file.", allow_abbrev = False)
    parser.add_argument("-i", "--input_bam",  type = str, required = True, help = "Input BAM file")
    parser.add_argument("-r", "--reference",  type = str, required = True, help = "Reference FASTA file")
    parser.add_argument("-o", "--orf_range",  type = str, required = True, help = "ORF range in the reference 0-based (e.g., '352-1383')")
    parser.add_argument("-p", "--prefix",     type = str, default = '',    help = "Output prefix")
    parser.add_argument("-b", "--base_qual",  type = int, default = 20,    help = "Minimum base quality")
    parser.add_argument("-c", "--chunk_size", type = int, default = 1000,  help = "Chunk size for processing reads")
    parser.add_argument("-t", "--threads",    type = int, default = 4,     help = "Number of threads")
    
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"Error: Unrecognized arguments: {' '.join(unknown)}", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    if args.prefix == '':
        prefix = os.path.splitext(os.path.basename(args.input_bam))[0]
    else:
        prefix = args.prefix

    # -- read input files -- #
    ref_dict = SeqIO.to_dict(SeqIO.parse(args.reference, "fasta"))

    orf_dict = {}
    codon_dict = {}
    orf_start, orf_end = map(int, args.orf_range.split('-'))

    # -- prepare output files -- #
    output_file = f"{prefix}.variant_counts.tsv"
    if os.path.exists(output_file):
        os.remove(output_file)

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Get the reference codons, please wait...", flush = True)
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

    # -- free memory -- #
    del ref_dict
    gc.collect()

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Get the base coverage, please wait...", flush = True)
    sub_region_size = (orf_end - orf_start + 1) // args.threads
    dict_base_cov = get_base_cov_in_chunk(args.input_bam, chrom, orf_start, orf_end, args.base_qual, sub_region_size, args.threads)

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Extracting read information, please wait...", flush = True)
    list_results = []
    for i, chunk_result in enumerate(read_bam_in_chunk(args.input_bam, args.orf_range, args.base_qual, args.chunk_size, args.threads)):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |--> Processed chunk {i+1} with {args.chunk_size} read pairs", flush = True)
        list_results.append(chunk_result)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |--> Finished.", flush = True)

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Creating the variant matrix, please wait...", flush = True)
    list_results_filtered = [df for df in list_results if df.height > 0]
    if list_results_filtered:
        df_variants = pl.concat(list_results_filtered, how = "vertical")
        df_variants_counts = ( df_variants.group_by("base_mut")
                                          .agg([pl.col("counts").sum().alias("counts"),
                                                pl.all().exclude(["base_mut", "counts"]).first()]) )
    
    # -- free memory -- #
    del list_results, list_results_filtered, df_variants
    gc.collect()

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Creating the output file, please wait...", flush = True)
    df_variants_counts = df_variants_counts.select([pl.col("counts"),
                                                    pl.col("base_cov_avg").alias("cov"),
                                                    pl.lit(0).alias("mean_length_variant_reads"),
                                                    pl.col("varying_bases"),
                                                    pl.col("base_mut"),
                                                    pl.col("varying_codons"),
                                                    pl.col("codon_mut"),
                                                    pl.col("aa_mut"),
                                                    pl.col("pos_mut")])
    df_variants_counts.write_csv(output_file, separator = "\t", null_value = "NA")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Done.", flush = True)
