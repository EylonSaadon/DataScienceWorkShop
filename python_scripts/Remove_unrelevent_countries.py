import csv
import sys
import os

PROJRCT_COUNTRIES = ['AUS','AUT','BEL','CAN','CHE','CHL','CZE','DEU','DNK','ESP','EST','FIN','FRA', 'GBR','GRC','HUN','IRL','ISL','ISR'
                     'ITA','JPN','KOR','LTU','LUX','LVA','MEX','NLD','NOR','NZL','POL','PRT','RUS','SVK','SVN','SWE','TUR','USA']

def remove_hexa_bytes(data):
    return data.decode('unicode_escape').encode('ascii', 'ignore')

def remove_countries(file_path):
    first_row = True
    f_handle = open(file_path, 'r')
    p = file_path.split('.')[0]
    new_file_path = p + '_new.csv'
    f_write_handle = open(new_file_path, 'wb')
    csv_reader = csv.reader(f_handle, delimiter=',')
    csv_writer = csv.writer(f_write_handle, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for row in csv_reader:
        if first_row:
            first_row = False
            fields = [remove_hexa_bytes(x) for x in row]
            csv_writer.writerow(fields)
            continue

        if check_valid_country(row[1]):
            csv_writer.writerow(row)

    f_handle.close()
    f_write_handle.close()


def check_valid_country(country_code):
    return country_code in PROJRCT_COUNTRIES


if __name__ == "__main__":
    '''
        receiving full file path
        '''
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        print 'Error must insert full file path, Exiting'
        sys.exit(0)

    if os.path.exists(file_path):
        remove_countries(file_path)

    print 'Done!'