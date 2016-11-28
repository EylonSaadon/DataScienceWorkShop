import csv
import sys
import os

def remove_hexa_bytes(data):
    return data.decode('unicode_escape').encode('ascii', 'ignore')

def export_indicator(file_path):
    first_row = True
    indicator_list = []
    count_indicators = 0
    f_handle = open(file_path,'r')
    p = file_path.split('.')[0]
    new_file_path = p+'_indicators.csv'
    f_write_handle = open(new_file_path, 'wb')
    csv_reader = csv.reader(f_handle,delimiter=',')
    csv_writer = csv.writer(f_write_handle, delimiter =',')

    for row in csv_reader:
        if first_row:
            first_row = False

            csv_writer.writerow[["Country","Indicator name"]]

            continue

        if row[2] not in indicator_list:
            count_indicators+=1
            indicator_list.append(row[2]+'\r\n')
            csv_writer.writerow([row[2]])
            f_write_handle.flush()


    f_handle.close()
    f_write_handle.close()
    print 'indicators number : {0}'.format(count_indicators)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        os.chdir(folder)

    #scan all csv files in current folder
    curr_path = os.getcwd()
    files = [file for file in os.listdir(curr_path) if file.endswith('csv')]
    for filename in files:
        file_path = os.path.join(curr_path,filename)
        print 'extract indicator from  file {0}'.format(filename)
        export_indicator(file_path)

    print 'Done!'