def my_code(x:float,y:float):
    try:
        return x/y
    except ZeroDivisionError:
        print('除数不能为0')
    except Exception as e:
        print(e)


def my_code_2(l):
    try:
        if len(l) == 0:
            return '列表无数据'
        if len(l) == 1:
            return l[0] / 2
        l = sorted(l)
        res = (l[0]+l[-1])/2
        return "{:.2f}".format(res)
    except ZeroDivisionError:
        print('除数不能为0')
    except Exception as e:
        print(e)


def my_code_3():
    import re
    import requests
    try:
        url = 'https://yf-circle.seeyouyima.com/v2/search_suggest'
        heard = {
            'myclient':'0120882000000000',
            'Authorization':'XDS 7.vEeBAubv9bVmTzAv1AESQYiGhPEF0p1GrI9ltoxBu90'
        }
        res = requests.get(url=url, headers=heard)

        results1 = re.findall(r'"id":\s*(\d+)', res.text)
        results2 = re.findall(r'"keyword":\s*"([\u4e00-\u9fa5]+)"', res.text)
        return results1,results2
    except Exception as e:
        print(e)


def _print_date(date,week):
    map_date = {
        '1': '星期一',
        '2': '星期二',
        '3': '星期三',
        '4': '星期四',
        '5': '星期五',
        '6': '星期六',
        '7': '星期天'
    }
    if week in ['6', '7']:
        msg = '节假日'
    else:
        msg = '工作日'
    print(f'{date}是{map_date[week]},是{msg}')


def my_code_4(date1, date2):
    from datetime import datetime,timedelta
    start_date = datetime.strptime(date1, "%Y-%m-%d")
    end_date = datetime.strptime(date2, "%Y-%m-%d")

    weekday1 = start_date.weekday() + 1
    weekday2 = end_date.weekday() + 1
    _print_date(date1,str(weekday1))
    _print_date(date2,str(weekday2))

    if start_date > end_date:
        start_date, end_date = end_date, start_date
    workdays = 0
    current_date = start_date
    while current_date <= end_date:
        week = current_date.weekday()
        if week < 5:
            workdays += 1
        current_date += timedelta(days=1)

    print(f'{date1}和{date2}之间的工作日相差是{workdays}')


if __name__ == '__main__':
    a='2024-12-01'
    b='2024-12-13'
    my_code_4(a, b)

    print(my_code_3())