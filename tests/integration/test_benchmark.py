import sdgym


def test_identity():
    output = sdgym.run(
        synthesizers=['Identity', 'Independent', 'Uniform'],
        datasets=['trains_v1', 'KRK_v1'],
    )

    assert not output.empty
    assert set(output['modality'].unique()) == {'single-table', 'multi-table'}

    scores = output.groupby('synthesizer').score.mean().sort_values()

    assert ['Uniform', 'Independent', 'Identity'] == scores.index.tolist()
