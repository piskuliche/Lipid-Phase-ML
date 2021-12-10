#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import argparse,pickle,sys,mllpa,os
from os.path import exists
from scipy import stats
from scipy.spatial import Voronoi, voronoi_plot_2d

parser = argparse.ArgumentParser()
parser.add_argument('-resnames',nargs='+',help='Residue Names')
parser.add_argument('-resnumbs',nargs='+',help='Residue Numbers')
parser.add_argument('-train', nargs='+',help='Training set names',required=True)
parser.add_argument('-phases', nargs='+',help='Names of phases (match training set order)',required=True)
parser.add_argument('-step', nargs='+', help='[0] Create Training Set, [1] Read in Configurations, [2] Analzye Configurations')
parser.add_argument('-data', default="system.data",type=str,help="Lammps data file to read")
parser.add_argument('-liptype',default="DPPC",nargs='+', type=str, help="Name of the lipid",required=True)
parser.add_argument('-nlipids', nargs='+', help="Number of lipids")
parser.add_argument('-skip',default=1, type=int, help="How many steps should be skipped")
parser.add_argument('-nblocks',default=5, type=int, help="Number of blocks for the calculation")
parser.add_argument('-low', default=0, type=int, help="Lower bound for voronoi plot")
parser.add_argument('-high', default=100, type=int, help="Higher bound for voronoi plot")
parser.add_argument('-rank', default=6, type=int, help="Rank for calculation of distances [default 6]")
args = parser.parse_args()

train      =   args.train
phases      =   args.phases
step        =   args.step
datafile    =   args.data
liptype     =   args.liptype
nlipids     =   args.nlipids
skip        =   args.skip
nblocks     =   args.nblocks
low         =   args.low
high        =   args.high
resnames    =   args.resnames
resnumbs    =   args.resnumbs
rank        =   args.rank

# Calculate t_val
t_val = stats.t.ppf(0.975,nblocks-1)/np.sqrt(nblocks)


### Functions ###

def run_checks():
    for lipid in liptype:
        if not exists(lipid+str(".names")):
            sys.exit("Error: %s.names does not exist"%lipid)


def step_train(lipid,N):
    tsets=[]
    print("Opening the training directories")
    print(train)
    for training_set in train:
        print(training_set)
        print(liptype)
        print(datafile)
        print("Using rank %s" % rank)
        if not exists("dumps/"+training_set):
            sys.exit("Error: dumps/"+training_set + " doesn't exist")
        if not exists(datafile):
            sys.exit("Error: %s not found" % datafile)
        tsets.append(mllpa.openSystem(datafile,datafile,lipid,trj="dumps/"+training_set,step=skip,rank=rank))
    final_model=mllpa.generateModel(tsets,phases,save_model=False)
    pickle.dump(final_model,open(lipid+"_model.pckl",'wb'),protocol=pickle.HIGHEST_PROTOCOL)
    print("Model dumped to %s_model.pckl" % lipid)
    print(final_model['scores']['final_score'])
    return

def step_read(lipid,N):
    temperatures = np.genfromtxt("temperatures.dat",usecols=0,dtype=str)
    allsystems={}
    tesselations={}
    for t in temperatures:
        allsystems[t]=mllpa.openSystem(datafile,datafile,lipid,trj="dumps/"+t+".lammpsdump",step=skip,rank=rank)
    pickle.dump(allsystems,open(lipid+"_allsystems.pckl",'wb'),protocol=pickle.HIGHEST_PROTOCOL)
    return

def step_classify(lipid,N):
    temperatures = np.genfromtxt("temperatures.dat",usecols=0,dtype=str)
    allsystems=pickle.load(open(lipid+"_allsystems.pckl",'rb'))
    final_model = pickle.load(open(lipid+"_model.pckl",'rb'))
    print("Recall - using model with these scores:")
    for key in final_model['scores']['final_score']:
        print("%s : %s" % (key, final_model['scores']['final_score'][key]))
    print("Now classifying!")
    allphases={}
    for t in temperatures:
        allphases[t]=allsystems[t].getPhases(final_model)
    pickle.dump(allphases,open(lipid+"_allphases.pckl",'wb'),protocol=pickle.HIGHEST_PROTOCOL)
    # re-writes the information on the class
    pickle.dump(allsystems,open(lipid+"_allsystems.pckl",'wb'),protocol=pickle.HIGHEST_PROTOCOL)
    return

def step_analyze(lipid):
    allphases=pickle.load(open(lipid+"_allphases.pckl",'rb'))
    temperatures = np.genfromtxt("temperatures.dat",usecols=0,dtype=float)
    fracs, errs = {},{}
    for p in phases:
        fracs[p]    = []
        errs[p]     = []
    for key in allphases:
        tmp_frac,tmp_err = calc_frac(allphases[key])
        for p in phases:
            fracs[p].append(tmp_frac[p])
            errs[p].append(tmp_err[p])
    for p in phases:
        np.savetxt(lipid+"_x"+p+".dat",np.c_[temperatures,fracs[p],errs[p]])
    pickle.dump(fracs,open(lipid+"_fracs.pckl",'wb'),protocol=pickle.HIGHEST_PROTOCOL)
    return

def calc_frac(phaseinfo):
    phaseinfo = phaseinfo[:np.shape(phaseinfo)[0]-(np.shape(phaseinfo)[0]%nblocks)]
    n={}
    bl_n={}
    for p in phases:
        n[p] = (phaseinfo==p)*1
        bl_n[p] = np.sum(np.split(n[p],nblocks),axis=2)
    bl_tot=np.sum(np.sum([bl_n[i] for i in phases],axis=0),axis=1)
    frac={}
    bl_frac={}
    err_frac={}
    for p in phases:
        frac[p] = np.sum(n[p])/np.sum([n[i] for i in phases])
        bl_frac[p]=np.divide(np.sum(bl_n[p],axis=1),bl_tot)
        err_frac[p]=np.std(bl_frac[p])*t_val
    return frac, err_frac

def single_frac(phaseinfo,N,target):
    n={}
    frac={}
    for p in phases:
        n[p] = (phaseinfo==p)*1
        frac[p]=np.divide(np.sum(n[p],axis=1),int(N))
    minerr, minframe = 1.0,0
    for i in range(len(frac[phases[0]])):
        test=frac[phases[0]][i]-target
        if test < minerr:
            minerr = test
            minframe=i
    return minframe


def step_Voronoi(lipid,N):
    temperatures = np.genfromtxt("temperatures.dat",usecols=0,dtype=str)
    allsystems=pickle.load(open(lipid+"_allsystems.pckl",'rb'))
    allphases=pickle.load(open(lipid+"_allphases.pckl",'rb'))
    fracs=pickle.load(open(lipid+"_fracs.pckl",'rb'))
    tesselations={}
    if not os.path.isdir("plots"): os.mkdir("plots")
    for t in temperatures:
        tesselations[t]=mllpa.doVoro([allsystems[t]],geometry='bilayer',read_neighbors=True)
    pickle.dump(tesselations,open(lipid+"_tesselations.pckl",'wb'),protocol=pickle.HIGHEST_PROTOCOL)
    return

def step_plot(lipid,N):
    temperatures = np.genfromtxt("temperatures.dat",usecols=0,dtype=str)
    allsystems=pickle.load(open(lipid+"_allsystems.pckl",'rb'))
    allphases=pickle.load(open(lipid+"_allphases.pckl",'rb'))
    tesselations=pickle.load(open(lipid+"_tesselations.pckl",'rb'))
    fracs=pickle.load(open(lipid+"_fracs.pckl",'rb'))
    itercnt=0
    for t in temperatures:
        frame=single_frac(allphases[t],N,fracs[phases[0]][itercnt])
        xy=tesselations[t].positions
        points=xy[frame,:,0:2]
        points = np.append(points,[[low-10,low-10],[low-10,high+10],[high+10,high+10],[high+10,low-10]],axis=0)
        vor = Voronoi(points)
        fig1, ax1 = plt.subplots(dpi=300,figsize=(4,4))
        fig = voronoi_plot_2d(vor,ax=ax1,show_vertices=False,line_colors='black',
                line_width=1, line_alpha=1, point_size=10)
        xs,ys = points.T
        ctmp=["blue","red","green","purple"]
        colors={}
        cnt = 0
        for p in phases:
            colors[p]=ctmp[cnt]
            cnt += 1
        for r in range(len(vor.point_region)):
            region = vor.regions[vor.point_region[r]]
            if not -1 in region:
                polygon = [vor.vertices[i] for i in region]
                plt.fill(*zip(*polygon), color=colors[allphases[t][frame][r]])
                plt.xlim([low,high]), plt.ylim([low,high])
        ax1.get_lines()[0].set_marker('o')
        ax1.get_lines()[0].set_markersize('2')
        ax1.get_lines()[0].set_markerfacecolor('black')
        ax1.get_lines()[0].set_markeredgecolor('black')
        plt.title(t)
        plt.xlabel('x (angstroms)')
        plt.ylabel('y (angstroms)')
        plt.xticks(np.arange(round(low), round(high)+5, 5))
        plt.yticks(np.arange(round(low), round(high)+5, 5))
        plt.savefig('plots/voronoi_'+t+'.png')
        plt.close()
        itercnt += 1
    return

### End Functions ###

run_checks()

if "setup" in step:
    print(resnames)
    print(resnumbs)
    cnt=0
    f=open('atom.names','w')
    g=open('data.residues','w')
    for res in resnames:
        atoms=np.genfromtxt(res+".names",dtype=str)
        if len(atoms.shape) == 0:
            atoms = np.array([atoms])
        for mol in range(int(resnumbs[cnt])):
            g.write("%s\n" % res)
            for a in atoms:
                f.write("%s\n" % a)
        cnt = cnt + 1
    f.close()
    g.close()


if "0" in step:
    print("Beginning to train model")
    cnt=0
    for lipid in liptype:
        step_train(lipid,nlipids[cnt])
        cnt += 1
if "1" in step:
    print("Reading in configurations")
    cnt = 0
    for lipid in liptype:
        step_read(lipid,nlipids[cnt])
        cnt += 1
if "2" in step:
    print("Finding Phases")
    cnt = 0
    for lipid in liptype:
        step_classify(lipid,nlipids[cnt])
        cnt += 1
if "3" in step:
    print("Analyzing Phase Info")
    for lipid in liptype:
        step_analyze(lipid)

if "4" in step:
    print("Calculating Voronoi")
    cnt=0
    for lipid in liptype:
        step_Voronoi(lipid,nlipids[cnt])
        cnt+=1

if "5" in step:
    print("Plotting")
    cnt=0
    for lipid in liptype:
        step_plot(lipid,nlipids[cnt])
