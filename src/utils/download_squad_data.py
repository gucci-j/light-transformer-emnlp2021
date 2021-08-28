"""Download SQuAD data."""

import urllib.request

from argparse import ArgumentParser
parser = ArgumentParser(description="Download SQuAD data")
parser.add_argument("-s", "--save_dir", help="(str) Where to save SQuAD data?", 
                    default="./squad_data")
args = parser.parse_args()

# SQuAD 1.0
train_ver1_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
dev_ver1_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
test_ver1_url = "https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py"

# SQuAD 2.0
train_ver2_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
dev_ver2_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
test_ver2_url = "https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/"

url_lists = (train_ver1_url, dev_ver1_url, test_ver1_url,
             train_ver2_url, dev_ver2_url, test_ver2_url)


def main():
    for url in url_lists:
        if url != test_ver2_url:
            urllib.request.urlretrieve(url, args.save_dir + "/" + url[url.rfind('/') + 1:])
        else:
            urllib.request.urlretrieve(url, args.save_dir + "/" + "evaluate-v2.0.py")


if __name__ == "__main__":
    main()