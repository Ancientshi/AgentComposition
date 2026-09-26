import json, pathlib, tempfile, unittest
from prepare_sft import sha_file
from train_sft import validate_data

class GuardTests(unittest.TestCase):
    def fixture(self,root,train_query='training query',valid_query='validation query',stale=False):
        test=root/'test_manifest.frozen.jsonl'
        test.write_text(json.dumps({'sample_id':'test1','query':'fixed test query'})+'\n')
        hashes={}
        for split,q in [('train',train_query),('valid',valid_query)]:
            row={'query':q,'completion':'stale' if stale else 'target','target':'target','input_ids':[1,2,3],
                 'labels':[-100,2,3],'attention_mask':[1,1,1]}
            p=root/f'sft_{split}.jsonl';p.write_text(json.dumps(row)+'\n');hashes[split]=sha_file(p)
        (root/'preparation_report.json').write_text(json.dumps({'test_manifest':str(test),'test_manifest_sha256':sha_file(test),'outputs_sha256':hashes}))
    def test_clean_data_passes(self):
        with tempfile.TemporaryDirectory() as d:
            r=pathlib.Path(d);self.fixture(r);_,v=validate_data(r,8192)
            self.assertEqual(v['train']['min_answer_tokens'],2)
    def test_test_leak_fails(self):
        with tempfile.TemporaryDirectory() as d:
            r=pathlib.Path(d);self.fixture(r,train_query='fixed  test\nquery')
            with self.assertRaisesRegex(ValueError,'test query'):validate_data(r,8192)
    def test_train_dev_leak_fails(self):
        with tempfile.TemporaryDirectory() as d:
            r=pathlib.Path(d);self.fixture(r,valid_query='training query')
            with self.assertRaisesRegex(ValueError,'Train/dev'):validate_data(r,8192)
    def test_stale_completion_fails(self):
        with tempfile.TemporaryDirectory() as d:
            r=pathlib.Path(d);self.fixture(r,stale=True)
            with self.assertRaisesRegex(ValueError,'Stale'):validate_data(r,8192)
    def test_modified_file_fails(self):
        with tempfile.TemporaryDirectory() as d:
            r=pathlib.Path(d);self.fixture(r)
            with (r/'sft_train.jsonl').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'changed'):validate_data(r,8192)
    def test_unreachable_tool_count_fails(self):
        with tempfile.TemporaryDirectory() as d:
            r=pathlib.Path(d);self.fixture(r)
            p=r/'sft_train.jsonl';x=json.loads(p.read_text())
            x['target']=x['completion']=' '.join(f'<<tool{i}>>' for i in range(7))
            p.write_text(json.dumps(x)+'\n')
            report=json.loads((r/'preparation_report.json').read_text());report['outputs_sha256']['train']=sha_file(p)
            (r/'preparation_report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'tool limit'):validate_data(r,8192)

if __name__=='__main__':unittest.main()
