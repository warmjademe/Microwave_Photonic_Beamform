"""两倍子集的成员选择与溯源验证；只用计划和临时元数据，不生成RF数据。"""
from collections import Counter
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from generate_dataset import plan_environments
from generate_native_dataset import digest,json_bytes,write_json
import resize_native_dataset as resize


class ResizeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows,_=plan_environments(17280,4320,20260925,'joint_stratified')
        cls.parent=dict(environments=rows,splits=dict(train=17280,test=4320),generation_fingerprint='parent-protocol')

    def test_exact_balance_and_original_split_seed_retained(self):
        rows=resize.select_rows(self.parent)
        self.assertEqual(len(rows),8640)
        for split,n,repeat in [('train',6912,32),('test',1728,8)]:
            selected=[r for r in rows if r['split']==split]
            count=Counter(r['joint_stratum_id'] for r in selected)
            self.assertEqual(len(selected),n);self.assertEqual(len(count),216)
            self.assertEqual(set(count.values()),{repeat})
        original={r['environment_id']:r for r in self.parent['environments']}
        self.assertTrue(all(r==original[r['environment_id']] for r in rows))
        self.assertEqual(len({r['seed'] for r in rows}),len(rows))
        self.assertTrue({r['environment_id'] for r in self.parent['environments'][:1581]}.issubset(
            {r['environment_id'] for r in rows}))

    def test_selection_ignores_quality_and_rejects_non_joint_counts(self):
        parent=deepcopy(self.parent);parent['received_quality']='must not be inspected'
        parent['completed_environments']='must not be inspected'
        self.assertEqual(resize.select_rows(parent),resize.select_rows(self.parent))
        with self.assertRaises(ValueError):resize.select_rows(parent,6913,1728)
        with self.assertRaises(ValueError):resize.select_rows(parent,17280+216,1728)

    def test_selection_provenance_and_member_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'provenance').mkdir()
            write_json(root/'provenance/parent_manifest.json',self.parent)
            rows=resize.select_rows(self.parent)
            selection=dict(selector_source_sha256=digest(Path(resize.__file__)),
                parent_manifest_sha256=digest(root/'provenance/parent_manifest.json'),
                selected_rows_sha256=hashlib.sha256(json_bytes(rows)).hexdigest())
            write_json(root/'selection.json',selection)
            manifest=dict(splits=dict(train=6912,test=1728),environments=rows,
                generation_fingerprint='parent-protocol',
                selection_fingerprint=hashlib.sha256(json_bytes(selection)).hexdigest())
            write_json(root/'manifest.json',manifest)
            self.assertTrue(resize.verify_selection(root)['passed'])
            manifest['environments'][0]['seed']+=1
            write_json(root/'manifest.json',manifest)
            with self.assertRaises(ValueError):resize.verify_selection(root)


if __name__=='__main__':unittest.main()
