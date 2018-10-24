from pathlib import Path

import requests
from tqdm import tqdm


def download(file_path, url):
    r = requests.get(url, stream=True)
    size = r.headers['content-length']
    size = int(int(size) / (1000 * 1000))
    with open(file_path, 'wb') as f:
        for chunk in tqdm(r.iter_content(chunk_size=1000 * 1000), total=size, unit="mb", desc=url):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def main(project_dir):

    years_dict = {
        2016: range(1, 7),
        2015: range(1, 13),
        2014: range(6, 13)
    }
    vendors = ["yellow"]    # additional options: "green", "fhv"

    total_files = len([month for months in years_dict.values() for month in months]) * len(vendors)
    file_number = 1

    for year in years_dict:
        for month in years_dict[year]:
            for vendor in vendors:

                print("Downloading taxi trip file {} of {}".format(file_number, total_files))
                file_number += 1

                file_name = "{}_tripdata_{}-{:02d}.csv".format(vendor, year, month)
                url = "https://s3.amazonaws.com/nyc-tlc/trip+data/{}".format(file_name)
                file_path = Path(project_dir / "data" / "raw" / file_name)
                if not file_path.is_file():     # check whether file already exists, to prevent unnecessary downloads
                    download(file_path, url)
                else:
                    print("File {} already exists. Skipping {}.".format(file_path, url))



if __name__ == '__main__':

    project_dir = Path(__file__).resolve().parents[2]

    main(project_dir)
