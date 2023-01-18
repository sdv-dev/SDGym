import sdgym
from sdgym.synthesizers import create_single_table_synthesizer


def test_identity():
    output = sdgym.benchmark_single_table(
        synthesizers=['DataIdentity', 'IndependentSynthesizer', 'UniformSynthesizer'],
        sdv_datasets=['student_placements'],
    )

    assert not output.empty
    assert 'Train_Time' in output
    assert 'Sample_Time' in output

    scores = output.groupby('Synthesizer').NewRowSynthesis.mean().sort_values()

    assert [
        'DataIdentity',
        'IndependentSynthesizer',
        'UniformSynthesizer',
    ] == scores.index.tolist()

    quality_scores = output.groupby('Synthesizer').Quality_Score.mean().sort_values()

    assert [
        'UniformSynthesizer',
        'IndependentSynthesizer',
        'DataIdentity',
    ] == quality_scores.index.tolist()


def get_trained_synthesizer_err(data, metadata):
    return {}


def sample_from_synthesizer_err(synthesizer, num_rows):
    raise ValueError('random error')


def test_error_handling():
    erroring_synthesizer = create_single_table_synthesizer(
        'my_synth', get_trained_synthesizer_err, sample_from_synthesizer_err)
    output = sdgym.benchmark_single_table(
        synthesizers=['DataIdentity', 'IndependentSynthesizer', 'UniformSynthesizer'],
        custom_synthesizers=[erroring_synthesizer],
        sdv_datasets=['student_placements'],
    )

    assert not output.empty
    assert 'Train_Time' in output
    assert 'Sample_Time' in output
    assert (
        output[output['Synthesizer'] == 'Custom:my_synth'][['Train_Time', 'Sample_Time']]
    ).isna().all(1).all()
