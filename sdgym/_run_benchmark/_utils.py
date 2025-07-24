from datetime import datetime


def get_run_name(date_str):
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f'Invalid date format: {date_str}. Expected YYYY-MM-DD.')

    return f'SDGym_results_{date.month:02d}_{date.day:02d}_{date.year}'
