from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from reconcile_super5 import read_fasta, reconcile
from refine import refine_file


class TestReconciliation(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.base=self.root/"base.fa"
        self.alternate=self.root/"alternate.fa"
        self.guide=self.root/"guide.fa"
        self.output=self.root/"out.fa"
        self.base.write_bytes(b">one label\nABCd-XYZ\n>two\nABC-wXYZ\n")
        self.alternate.write_bytes(b">two\nABCWXYZ\n>one label\nABCDXYZ\n")
        self.guide.write_bytes(self.alternate.read_bytes())

    def test_supported_local_change_and_original_format_contract(self):
        self.assertEqual(reconcile(self.base,self.alternate,[self.guide],self.output,"local"),1)
        self.assertEqual(list(read_fasta(self.output).items()),
                         [(b"one label",b"ABCdXYZ"),(b"two",b"ABCwXYZ")])
        self.guide.write_bytes(self.base.read_bytes())
        self.assertEqual(reconcile(self.base,self.alternate,[self.guide],self.output,"local"),0)
        self.assertEqual(self.output.read_bytes(),self.base.read_bytes())

    def test_changed_residues_and_input_output_alias(self):
        with self.assertRaises(ValueError):
            reconcile(self.base,self.alternate,[self.guide],self.base,"local")
        self.output.write_text("previous")
        self.guide.write_bytes(self.guide.read_bytes().replace(b"ABCD",b"AXCD"))
        with self.assertRaises(ValueError):
            reconcile(self.base,self.alternate,[self.guide],self.output,"local")
        self.assertEqual(self.output.read_text(),"previous")

    def test_duplicate_labels(self):
        self.guide.write_bytes(b">two\nABC\n>two\nABC\n")
        with self.assertRaises(ValueError):
            reconcile(self.base,self.alternate,[self.guide],self.output,"local")

    def test_pair_preserving_merge_and_source_protection(self):
        stats = refine_file(self.base, self.guide, self.output)
        self.assertEqual(stats["new_pairs"], 1)
        self.assertEqual(list(read_fasta(self.output).items()),
                         [(b"one label", b"ABCdXYZ"), (b"two", b"ABCwXYZ")])
        with self.assertRaises(ValueError):
            refine_file(self.base, self.guide, self.guide)

    def test_pair_rule_with_computed_alternative(self):
        self.assertEqual(reconcile(self.base, self.alternate, [self.guide], self.output, "pair"), 1)
        self.assertEqual(read_fasta(self.output)[b"one label"], b"ABCdXYZ")


if __name__=="__main__": unittest.main()
