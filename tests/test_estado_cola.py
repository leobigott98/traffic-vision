from estado_cola import EstimadorCola


def obs(ident=1):
    return dict(track_id=ident,punto_xy=[50.,50.],xyxy=[40.,20.,60.,50.],asignacion={'zona_id':'z'})


def test_queue_requires_history_and_missing_frame_clears_it():
    e=EstimadorCola(['z'])
    counts={'z':dict(total=1,fraccion_visible=1.)}
    for i in range(45):
        result=e.actualizar(i,i/10,[obs()],counts,True)['z']
        if i==0:assert result['indeterminados']==1 and result['cola_visible_proxy']==0
    assert result['cola_visible_proxy']==1
    e.actualizar(45,4.5,[],{'z':dict(total=0,fraccion_visible=1.)},True)
    result=e.actualizar(46,4.6,[obs()],counts,True)['z']
    assert result['cola_visible_proxy']==0 and result['indeterminados']==1
