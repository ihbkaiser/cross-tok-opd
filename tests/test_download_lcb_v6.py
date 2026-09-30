import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('download_lcb',Path(__file__).resolve().parents[1]/'scripts/evaluation/download_lcb_v6.py')
M=importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class Response(io.BytesIO):
    def __init__(self,data,status,headers):
        super().__init__(data)
        self.status,self.headers=status,headers


class DownloadTests(unittest.TestCase):
    def test_resume_verified_bytes(self):
        data=b'complete source bytes'
        class Opener:
            def open(self,req,timeout):
                self.req=req
                return Response(data[5:],206,{'Content-Range':f'bytes 5-{len(data)-1}/{len(data)}'})
        opener=Opener()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'test.part').write_bytes(data[:5])
            M.download(opener,root,'test',len(data),hashlib.sha256(data).hexdigest())
            self.assertEqual((root/'test').read_bytes(),data)
            self.assertEqual(opener.req.get_header('Range'),'bytes=5-')

    def test_server_ignoring_range_restarts_partial_only(self):
        data=b'correct'
        class Opener:
            def open(self,req,timeout): return Response(data,200,{})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'test.part').write_bytes(b'bad')
            M.download(Opener(),root,'test',len(data),hashlib.sha256(data).hexdigest())
            self.assertEqual((root/'test').read_bytes(),data)
            with self.assertRaises(ValueError): M.download(Opener(),root,'test',len(data),'0'*64)

    def test_functional_contract_validation(self):
        row={'question_id':'1','question_content':'q','starter_code':'class Solution:',
             'public_test_cases':json.dumps([{'testtype':'functional','input':'1','output':'2'}]),
             'private_test_cases':'preserved-opaque-data','metadata':'{"func_name":"f"}'}
        self.assertEqual(M.validate_row(row),('1',True))
        row['metadata']='{}'
        with self.assertRaises(ValueError):M.validate_row(row)


if __name__=='__main__':unittest.main()
