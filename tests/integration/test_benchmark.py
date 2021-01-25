import sdgym


def test_identity():
    output = sdgym.run(
        synthesizers=['Identity', 'Uniform'],
        datasets=['got_families', 'KRK_v1'],
    )

    assert not output.empty
    assert set(output['modality'].unique()) == {'single-table', 'multi-table'}
    identity_score = output[output.synthesizer == 'Identity'].score.mean()
    uniform_score = output[output.synthesizer == 'Uniform'].score.mean()
    assert identity_score > uniform_score
