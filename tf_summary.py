import tensorflow as tf
import argparse, csv, os


def process_events_file(file_path):
    # Your logic to process each events file goes here
    print(f"Processing file: {file_path}")

def iterate_through_files(directory_path):
    # Iterate through all files in a directory and its subdirectories
    file_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.startswith('events.out') and file.endswith('.0'):
                file_path = root + '/' + file
                process_events_file(file_path)
                file_paths.append(file_path)

    return file_paths

def main(args):

    # Find every file that starts with events.out in subdirectories of args.summary_path
    file_paths = iterate_through_files(args.summary_path)

    # Create summary_iterator
    for file_path in file_paths:
        summary_iterator = tf.compat.v1.train.summary_iterator(file_path)

        # Iterate through event
        epoch = []
        t_loss = []
        v_loss = []
        acc = []
        c_above90 = []

        for event in summary_iterator:
            # Print event
            #print(event)
            # Iterate through summary
            for value in event.summary.value:
                # Print value
                #print(value)
                # Print only if it has a tag
                if value.tag:
                    # Print tag and corresponding value
                    if value.tag == 'train/loss':
                        t_loss.append(value.simple_value)
                    if value.tag == 'eval/loss':
                        v_loss.append(value.simple_value)
                    if value.tag == 'eval/accuracy':
                        acc.append(value.simple_value)
                    if value.tag == 'eval/c_above90':
                        c_above90.append(value.simple_value)
                    if value.tag == 'train/epoch':
                        if len(epoch) == 0 or value.simple_value != epoch[-1]:
                            epoch.append(value.simple_value)
                    
                    #print(value.tag, value.simple_value)

        # Save epoch, loss, acc, c_above_90 to csv
        
        csv_file_name = '_'.join(file_path.split('/')[-1].split('.')[:-1]) + '.csv'
        file_dir = '/'.join(file_path.split('/')[:-1]) + '/'
        with open(file_dir + csv_file_name, mode='w', newline = '') as csv_file:
            fieldnames = ['epoch', 't_loss', 'v_loss', 'acc', 'c_above_90']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            for i in range(len(epoch)):
                writer.writerow({'epoch': epoch[i] if len(epoch) > i else '', 
                                't_loss': t_loss[i] if len(t_loss) > i else '', 
                                'v_loss': v_loss[i] if len(v_loss) > i else '', 
                                'acc': acc[i] if len(acc) > i else '', 
                                'c_above_90': c_above90[i] if len(c_above90) > i else ''})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_path', type=str, default=None)
    args = parser.parse_args()
    main(args)