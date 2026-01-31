import re
import pandas as pd


def get_date_range(df):
    start = df['date'].min().strftime("%d %b %Y")
    end = df['date'].max().strftime("%d %b %Y")
    return start, end

def preprocess(data):
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[APap][Mm]\s-\s'

    messages = re.split(pattern, data)[1:]

    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    df['message_date'] = (
        df['message_date']
        .str.replace('\u202f', ' ', regex=False)
        .str.replace(' -', '', regex=False)
        .str.strip()
    )
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %I:%M %p')

    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    time_period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            time_period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            time_period.append(str('00') + "-" + str(hour + 1))
        else:
            time_period.append(str(hour) + "-" + str(hour + 1))

    df['time_period'] = time_period
    return df
