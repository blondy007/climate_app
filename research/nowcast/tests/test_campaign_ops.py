"""Safety properties of the second trial's scheduling, gates and isolation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('campaign',Path(__file__).resolve().parents[1] / 'campaign_ops.py')
ops=importlib.util.module_from_spec(spec); spec.loader.exec_module(ops)


def source(reason=None):
    return {'frames':[{'product':'DBZH','age_minutes':20,
                       'sites':{'los-alcazares':{'nearby_coverage':1}}} for _ in range(3)]+
                     [{'product':'RATE','age_minutes':25}],
            'sites':{'los-alcazares':{'can_issue':reason is None,'reason':reason}}}


class CampaignSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.patches=[patch.object(ops,'ROOT',self.root),patch.object(ops,'now',return_value=ops.START),
                      patch.object(ops,'probe',return_value=source()),
                      patch.object(ops.shutil,'disk_usage',return_value=type('Disk',(),{'free':20*1024**3})()),
                      patch.object(ops,'run',return_value=type('Result',(),{'stdout':'test-container-id'})())]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)

    def test_no_early_late_or_deadline_launch(self):
        for moment in (ops.START.replace(hour=10),ops.START.replace(minute=35),ops.END):
            with self.subTest(moment=moment),patch.object(ops,'now',return_value=moment):
                with self.assertRaisesRegex(RuntimeError,'Outside'):
                    ops.start()
        ops.probe.assert_not_called(); ops.run.assert_not_called()

    def test_never_recreates_or_resets_existing_session(self):
        for filename in ('live-config.json','live-budget.json'):
            p=self.root/filename; p.write_text('original')
            with self.assertRaisesRegex(RuntimeError,'never be recreated'):
                ops.start()
            self.assertEqual(p.read_text(),'original'); p.unlink()
        ops.probe.assert_not_called()

    def test_source_failures_are_preserved_and_retry_count_is_bounded(self):
        ops.probe.return_value=source('Insufficient radar coverage')
        for n in (1,2,3):
            with self.assertRaisesRegex(RuntimeError,'source gate failed'):
                ops.start()
            self.assertTrue((self.root/f'start-blocked-{n}.json').exists())
        with self.assertRaisesRegex(RuntimeError,'three bounded'):
            ops.start()
        self.assertEqual(ops.probe.call_count,3)
        ops.run.assert_not_called()
        self.assertFalse((self.root/'live-budget.json').exists())

    def test_stale_and_uncovered_data_never_start(self):
        for kind in ('DBZH','RATE','coverage'):
            value=source()
            if kind == 'coverage':
                value['frames'][2]['sites']['los-alcazares']['nearby_coverage']=0
            else:
                for row in value['frames']:
                    if row['product']==kind: row['age_minutes']=90
            ops.probe.return_value=value
            with self.assertRaisesRegex(RuntimeError,'source gate failed'):
                ops.start()
        ops.run.assert_not_called()

    def test_resource_and_data_isolation_and_probe_budget_are_retained(self):
        ops.start()
        args=ops.run.call_args_list[0].args[0]
        self.assertEqual(args[:2],['docker','create'])
        self.assertIn(f'type=bind,src={self.root},dst=/data',args)
        self.assertIn('--read-only',args)
        self.assertNotIn('--publish',args)
        self.assertEqual(args[args.index('--memory')+1],'768m')
        self.assertEqual(args[args.index('--cpus')+1],'0.5')
        self.assertEqual(args[-1],ops.END.isoformat())
        self.assertEqual(json.loads((self.root/'live-budget.json').read_text())['reserved_download_bytes'],750_000_000)
        calls=ops.run.call_count
        ops.start()
        self.assertEqual(ops.run.call_count,calls)

    def test_finish_cannot_stop_before_deadline(self):
        with self.assertRaisesRegex(RuntimeError,'before the authorized deadline'):
            ops.finish()
        ops.run.assert_not_called()

    def test_finish_targets_only_new_collector(self):
        with patch.object(ops,'now',return_value=ops.END),patch.object(ops.subprocess,'run',
                return_value=type('Result',(),{'returncode':0,'stdout':'true'})()):
            ops.finish()
        self.assertEqual(ops.run.call_args.args[0],['docker','stop','--time','20',ops.NAME])
        self.assertTrue((self.root/'deadline-stop.json').exists())


if __name__ == '__main__':
    unittest.main()
