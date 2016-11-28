import csv
import sys
import os

def remove_hexa_bytes(data):
    return data.decode('unicode_escape').encode('ascii', 'ignore')

def remove_empty_lines(file_path):
    first_row = True
    f_handle = open(file_path,'r')
    p = file_path.split('.')[0]
    new_file_path = p+'_new.csv'
    f_write_handle = open(new_file_path, 'wb')
    csv_reader = csv.reader(f_handle,delimiter=',')
    csv_writer = csv.writer(f_write_handle,delimiter=',',quoting=csv.QUOTE_MINIMAL)
    for row in csv_reader:
        if first_row:
            first_row = False
            fields = [remove_hexa_bytes(x) for x in row]
            csv_writer.writerow(fields)
            continue

        if not check_sample_is_empty(row[4:]):
            csv_writer.writerow(row)


    f_handle.close()
    f_write_handle.close()

def check_sample_is_empty(sample_row):
    '''
    check if the sample row is empty - fill with empty strings
    :param sample_row:
    :return:
    '''
    empty_strings = [""]*len(sample_row)
    return empty_strings == sample_row



if __name__ == "__main__":
    '''
    receiving full file path
    '''
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        print 'Error must insert full file path, Exiting'
        sys.exit(0)


    # #scan all csv files in current folder
    # curr_path = os.getcwd()
    # files = [file for file in os.listdir(curr_path) if file.endswith('csv') and file.count('new') == 0]
    # for filename in files:
    #     file_path = os.path.join(curr_path,filename)
    #     print 'remove empty lines from file {0}'.format(filename)
    #     remove_empty_lines(file_path)
    if os.path.exists(file_path):
        remove_empty_lines(file_path)

    print 'Done!'