import os


def reform_date_str(date_str):
    lst = date_str.split('/')
    year = lst[-1]
    day = lst[-2]
    month = lst[0]
    if len(day) == 1:
        day = '0' + day
    if len(month) == 1:
        month = '0' + month
    return '20' + year + month + day