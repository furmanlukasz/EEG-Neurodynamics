import Neurodynamics
from argparse import ArgumentParser
from pyrqa.image_generator import ImageGenerator
from numpy import genfromtxt

parser = ArgumentParser()
parser.add_argument('eidx',type=int)
args = parser.parse_args()

if __name__ == '__main__':
    print(("#" * 50) + ' PYRQA COMPUTATION STARTED ' + ("#" * 50))
    sig = genfromtxt('Temp/sft' + str(args.eidx) + '/dist.csv', delimiter=',')
    task = Neurodynamics.ComputeTask('testSubject',sig[:5024],250,0)
    re = task.computeRP(td=0, emb=1, metric='Cosine')

    ImageGenerator.save_recurrence_plot(re,'Temp/pyrqa'+str(args.eidx)+'/rp_plot.jpg')
    print(("# " * 30) + ' PYRQA COMPUTATION FINISHED ' + (" #" * 30))