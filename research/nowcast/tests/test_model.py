from datetime import datetime, timedelta, timezone
from collections import Counter
import time
import unittest

import numpy as np

from research.nowcast.model import availability, advect, categorical, forecast, rate_grid_velocity, time_steps
from research.nowcast.replay import choose_sources, summarize


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.start=datetime(2026,10,20,tzinfo=timezone.utc)
        self.issue=self.start+timedelta(minutes=15)
        self.site="test"
        def metadata(when,grid,shape):
            return {"nominal_end":when.isoformat(),"retrieved_at":(when+timedelta(seconds=1)).isoformat(),
                    "catalog_last_modified":(when+timedelta(seconds=1)).isoformat(),
                    "observation_start":(when-timedelta(minutes=10)).isoformat(),"sha256_crop":"a"*64,
                    "projection":"same","global_origin_xy":[0,0],"global_shape":[shape,shape],"grid_km":grid,
                    "sites":{self.site:{"shape":[shape,shape],"centre":[shape//2,shape//2],"crop_origin_rc":[0,0]}}}
        self.meta=[metadata(self.start+timedelta(minutes=n),1,64) for n in (0,5,10)]
        self.rate_meta=metadata(self.start,2,32)

    def test_advection_includes_observation_age_in_forecast_horizon(self):
        field=np.zeros((32,32))
        field[16,11]=1
        velocity=np.stack([np.ones_like(field),np.zeros_like(field)])
        issue=self.start+timedelta(minutes=10)
        result=advect(field,velocity,self.start,issue,(15,))
        self.assertAlmostEqual(result[0,16,16],1)
        self.assertEqual(time_steps(self.start,issue,(15,)),[5])
        self.assertTrue(np.isnan(result[0,16,0]))

    def test_actual_live_time_is_rounded_forward_for_targets_only(self):
        issue=self.start+timedelta(seconds=1)
        self.assertEqual(time_steps(self.start,issue,(15,)),[4])

    def test_future_and_unpublished_observations_are_rejected(self):
        with self.assertRaises(ValueError):
            time_steps(self.issue,self.start,(15,))
        self.meta[-1]["retrieved_at"]=(self.issue+timedelta(seconds=1)).isoformat()
        with self.assertRaisesRegex(ValueError,"available"):
            forecast([np.zeros((64,64))]*3,self.meta,np.zeros((32,32)),self.rate_meta,self.site,self.issue)

    def test_catalog_publication_controls_retrospective_selection(self):
        old={**self.meta[-1],"nominal_end":(self.start+timedelta(minutes=5)).isoformat(),
             "catalog_last_modified":(self.start+timedelta(minutes=7)).isoformat()}
        late={**self.meta[-1],"catalog_last_modified":(self.issue+timedelta(minutes=1)).isoformat()}
        self.assertGreater(availability(late,True),self.issue)
        self.assertLess(availability(old,True),self.issue)

    def test_replay_chooses_only_already_published_input_sequence(self):
        # Use explicit keys to keep the assertion independent of timestamp formatting.
        dbzh={key:meta for key,meta in zip(("20261020T0000","20261020T0005","20261020T0010"),self.meta)}
        dbzh["20261020T0015"]={**self.meta[-1],"nominal_end":self.issue.isoformat(),
                               "catalog_last_modified":(self.issue+timedelta(minutes=1)).isoformat()}
        keys,rate=choose_sources(dbzh,{"20261020T0000":self.rate_meta},self.issue,True)
        self.assertEqual(keys,["20261020T0000","20261020T0005","20261020T0010"])
        self.assertEqual(rate,"20261020T0000")

    def test_grid_resampling_conserves_physical_motion(self):
        velocity=np.ones((2,64,64))*2
        resampled=rate_grid_velocity(velocity,self.meta[-1],self.rate_meta,self.site)
        np.testing.assert_allclose(resampled,1)

    def test_missing_or_stale_radar_never_means_dry(self):
        rate=np.zeros((32,32));rate[16,16]=np.nan
        with self.assertRaisesRegex(ValueError,"coverage"):
            forecast([np.zeros((64,64))]*3,self.meta,rate,self.rate_meta,self.site,self.issue)
        with self.assertRaisesRegex(ValueError,"Stale"):
            forecast([np.zeros((64,64))]*3,self.meta,np.zeros((32,32)),self.rate_meta,self.site,self.issue+timedelta(hours=1))

    def test_all_dry_cases_cannot_claim_positive_event_skill(self):
        result=categorical([{"rain_rate":0,"observed_rate":0}]*100)
        self.assertIsNone(result["csi"])
        self.assertIsNone(result["probability_of_detection"])
        self.assertEqual(result["correct_dry"],100)

    def test_false_alarms_misses_and_unknowns_are_distinct(self):
        result=categorical([{"rain_rate":1,"observed_rate":1},{"rain_rate":1,"observed_rate":0},
                            {"rain_rate":0,"observed_rate":1},{"rain_rate":None,"observed_rate":0}])
        self.assertEqual([result[key] for key in ("hits","misses","false_alarms","unavailable")],[1,1,1,1])
        self.assertAlmostEqual(result["csi"],1/3)

    def test_dry_forecast_remains_experimental(self):
        result=forecast([np.zeros((64,64))]*3,self.meta,np.zeros((32,32)),self.rate_meta,self.site,self.issue)
        self.assertFalse(result["approved_for_public_use"])
        self.assertEqual([point["rain_rate"] for point in result["points"]],[0,0,0])

    def test_report_counts_rejections_and_compares_identical_pairs(self):
        rows=[{"site":"los-alcazares","split":"validation","lead_minutes":15,
               "forecast_emitted":emitted,"rain_rate":predicted,"persistence_rate":1,"observed_rate":1}
              for emitted,predicted in ((True,1),(False,None),(True,None))]
        report=summarize(rows,Counter(),Counter({("los-alcazares","validation"):3}),time.monotonic(),1)
        value=next(row for row in report["results"] if row["site"]=="los-alcazares"
                   and row["split"]=="validation" and row["lead_minutes"]==15)
        self.assertEqual(value["paired_cases"],1)
        self.assertEqual(value["attempted_origins"],3)
        self.assertAlmostEqual(value["availability"],2/3)
        self.assertEqual(value["nowcast_all"]["unavailable"],2)
        self.assertEqual(value["nowcast"]["hits"],value["persistence"]["hits"])


if __name__=="__main__":
    unittest.main()
