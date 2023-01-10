## Reading Line by Line 

import os

INPUT_FILE_PATH = "/Volumes/NO NAME/ntt-data/test/01_総数_20190101_20190131_sample.csv"
OUTPUT_FILE_PATH = "./output.csv"


print(f'File Size is {os.stat(INPUT_FILE_PATH).st_size / (2 ** 30)} GB')

# txt_file = open(INPUT_FILE_PATH)

# count = 0

# for line in txt_file:
#     # we can process file line by line here, for simplicity I am taking count of lines
#     count += 1

# txt_file.close()

# print(f'Number of Lines in the file is {count}')

# print('Peak Memory Usage =', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
# print('User Mode Time =', resource.getrusage(resource.RUSAGE_SELF).ru_utime)
# print('System Mode Time =', resource.getrusage(resource.RUSAGE_SELF).ru_stime)

## Example using dask

import time

import dask.dataframe as dd


def sample_dask():
    start = time.time()

    # Read the .csv.gz file using Dask's read_csv function
    df = dd.read_csv(INPUT_FILE_PATH)
    print(f"reading step completed in {time.time() - start} seconds")

    # convert to string
    df['area'] = df['area'].astype(str)
    print(f"string conversion step completed in {time.time() - start} seconds")

    # Filter the dataframe to only include rows where the "area" column starts with 5339
    df_filtered = df[df['area'].str.startswith('5339')]

    # Output the filtered dataframe to a new CSV file
    df_filtered.to_csv('filtered.csv')
    print(f"filtering step completed in {time.time() - start} seconds")

    end = time.time()

    print(f"dask took {end - start} seconds to finish")

# Reading line by line using gzip 

import gzip

def sample_gzip():
    start = time.time()

    # Open the .csv.gz file in read mode using gzip
    with gzip.open(INPUT_FILE_PATH, 'rt') as f:
        # Open the output file in write mode
        with open('filtered2.csv', 'w') as out:
            # Iterate over the lines in the .csv.gz file
            for line in f:
                # Split the line into columns using the comma delimiter
                columns = line.split(',')
                # Check if the 5th column starts with 5339
                if columns[3].startswith('5339'):
                    # Write the line to the output file
                    out.write(line)

    end = time.time()

    print(f"gzip took {end - start} seconds to finish")

## datatable implementation 

# import datatable as dt
# from datatable import f
# def sample_datatable():
#     start = time.time()

#     # Read the .csv.gz file using datatable's fread function
#     df = dt.fread(INPUT_FILE_PATH)
#     print(f"reading step completed in {time.time() - start} seconds")

#     # Filter the dataframe to only include rows where the "area" column starts with 5339
#     df_filtered = df[(533900000 <= f.area) & (f.area <= 534000000), :]
#     print(f"filtering step completed in {time.time() - start} seconds")

#     # Output the filtered dataframe to a new CSV file
#     df_filtered.to_hdf('./filtered3.h5', key='stage', mode='w')

#     end = time.time()

#     print(f"datatable took {end - start} seconds to finish")

def main():
    files = []



if __name__=="__main__":
    main()