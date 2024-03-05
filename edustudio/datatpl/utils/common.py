from contextlib import closing
from tqdm import tqdm
import requests
import zipfile


class BigfileDownloader(object):
    @staticmethod
    def download(url, title, filepath, chunk_size=10240):
        with closing(requests.get(url, stream=True, allow_redirects=True)) as resp:
            if resp.status_code != 200:
                raise Exception("[ERROR]: {} - {} -{}".format(str(resp.status_code), title, url))
            chunk_size = chunk_size
            content_size = int(resp.headers['content-length'])  
            with tqdm(total=content_size, desc=title, ncols=100) as pbar:
                with open(filepath, 'wb') as f:
                    for data in resp.iter_content(chunk_size=chunk_size):
                        f.write(data)
                        pbar.update(len(data))


class DecompressionUtil(object):
    @staticmethod
    def unzip_file(zip_src, dst_dir):
        r = zipfile.is_zipfile(zip_src)
        if r:     
            fz = zipfile.ZipFile(zip_src, 'r')
            for file in tqdm(fz.namelist(), desc='unzip...', ncols=100):
                fz.extract(file, dst_dir)       
        else:
            raise Exception(f'{zip_src} is not a zip file')
