import sdgym


def test_identity():
    output = sdgym.benchmark_single_table(
        synthesizers=['DataIdentity', 'IndependentSynthesizer', 'UniformSynthesizer'],
        sdv_datasets=['student_placements'],
    )

    assert not output.empty
    assert set(output['modality'].unique()) == {'single-table'}
    assert 'train_time' in output
    assert 'sample_time' in output

    scores = output.groupby('synthesizer').score.mean().sort_values()

    assert [
        'DataIdentity',
        'IndependentSynthesizer',
        'UniformSynthesizer',
    ] == scores.index.tolist()
