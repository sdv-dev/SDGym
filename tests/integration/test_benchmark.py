import sdgym


def test_identity():
    output = sdgym.run(
        synthesizers=['Identity', 'Uniform'],
        datasets=['got_families', 'KRK_v1'],
    )

    assert not output.empty
    assert set(output['modality'].unique()) == {'single-table', 'multi-table'}
    assert output[output.synthesizer == 'Identity'].score.mean() > 0.9
    assert output[output.synthesizer == 'Uniform'].score.mean() < 0.8
