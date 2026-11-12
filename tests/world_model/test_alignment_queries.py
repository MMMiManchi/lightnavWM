import math

import numpy as np

from experiments.world_model.scene_memory import PersistentSceneMemory


def test_verified_transport_is_used_without_overwriting_recorded_direction(tmp_path):
    bank=PersistentSceneMemory(tmp_path/'memory.sqlite',contract_id='fixed',feature_dim=4)
    old=np.array([math.sin(.8),math.cos(.8)])
    now=np.array([math.sin(-.4),math.cos(-.4)])
    bank.record_observed(context='house',episode_id='old',event_id='actual',source_observation='past',target_observation='outcome',
        execution_ref='real',source_feature=[1,0,0,0],observed_future=[0,1,0,0],branch_direction=old,executed=True,arrived=True)
    kwargs=dict(context='house',before_sequence=1,source_feature=[1,0,0,0],predicted_future=[0,1,0,0],
                branch_direction=now,exclude_episode='current',query_kind='source')
    assert not bank.retrieve(**kwargs)
    assert not bank.retrieve(**kwargs,direction_transform=lambda *_:None)
    result=bank.retrieve(**kwargs,direction_transform=lambda *_:(now,{'method':'tested_geometry'}))
    assert len(result)==1
    assert np.allclose(result[0].branch_direction,old)
    assert np.allclose(result[0].aligned_branch_direction,now)
    assert result[0].alignment_info['method']=='tested_geometry'
    raw=bank.connection.execute('select branch from experiences').fetchone()[0]
    assert np.allclose(np.frombuffer(raw,dtype='<f4'),old)
    bank.close()
