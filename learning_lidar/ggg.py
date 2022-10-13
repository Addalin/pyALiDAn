from multiprocessing import Pool
import tqdm
import time

def _foo(my_number):
   square = my_number * my_number
   time.sleep(1)
   return square

if __name__ == '__main__':
   with Pool(4) as p:
      for i in tqdm.tqdm(p.imap(_foo, range(30)), total=30):
          pass