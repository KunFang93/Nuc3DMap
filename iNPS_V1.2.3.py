#!/usr/bin/env python
#--coding:utf-8 --


#################################################################################################
#################################################################################################
########                                                                                 ########
########    An improved Algorithm for Nucleosome Positioning from Sequencing Data        ########
########                                                                                 ########
########    Program to position nucleosomes with Keji Zhao's tag coodinate bed files.    ########
########                                                                                 ########
########    Author:  CHEN Weizhong                                                       ########
########                                                                                 ########
########    Working Environment:  Python3                                                ########
########                                                                                 ########
########    Date:      2012-05-02                                                        ########
########                                                                                 ########
########    Modified:  2013-11-15                                                        ########
########                                                                                 ########
########    Modified:  2014-02-24  Version 1.0 , Add Poisson test                        ########
########                                                                                 ########
########    Update:    2014-10-25  Version 1.1                                           ########
########                                                                                 ########
########    Update:    2014-11-13  Version 1.1.2                                         ########
########                                                                                 ########
########    Update:    2015-01-16  Version 1.2.0 , Available for paired-end data         ########
########                                                                                 ########
########    Update:    2015-03-03  Version 1.2.1                                         ########
########                                                                                 ########
########    Update:    2015-08-21  Version 1.2.2                                         ########
########                                                                                 ########
#################################################################################################
#################################################################################################








### Input and output files parameters
FilesParameters={'inputfile_for_nucleosome_positioning':'none',
                 'Chosen_Chromosome_Abbreviation':'none',
                 'Chosen_Chromosome_Length':'none',
                 'outputfile_like_bed':'',
                 'outputfile_like_wig':'',
                 }




### The standard deviation in Gaussian convolution ### Please set them with integers.
PreliminaryConvolutionParameters={'sigma':3,
                                  'times_of_sigma':3,
                                  'secondary_sigma':1}

### Threshold of height and width of peak
threshold={'PCC_independence_1':0.8,   'concavity_and_convexity_1':0.8,  'gap_length_1':100,  'height_ratio_1':0,
           'PCC_independence_2':0.2,   'concavity_and_convexity_2':0.8,  'gap_length_2':10,   'height_ratio_2':0.2,
           'PCC_independence_3':-0.5,  'concavity_and_convexity_3':0.8,  'gap_length_3':6,    'height_ratio_3':0.4,
           'PCC_independence_4':-1,    'concavity_and_convexity_4':0.8,  'gap_length_4':20,   'height_ratio_4':0.5,
           'PCC_independence_5':-1,    'concavity_and_convexity_5':0,    'gap_length_5':6,    'height_ratio_5':0.7,
           'PCC_independence_6':-1,    'concavity_and_convexity_6':0,    'gap_length_6':10,   'height_ratio_6':0.8,
           ########
           'influence_coefficient_cutoff':0,  'influence_coefficient_absolute_cutoff':0,
           'influence_coefficient_distance':9,  'influence_coefficient_distance_center':13,
           ########
           'filter_switch':'on',  'merging_center_distance':150,  'merging_height_watershed':5,
           'merging_height_ratio_1':0.80,  'merging_gap_1':50,  'merging_percentage_1':0.75,
           'merging_height_ratio_2':0.80,  'merging_gap_2':70,  'merging_percentage_2':0.75,
           'merging_height_ratio_3':0.90,  'merging_gap_3':70,  'merging_percentage_3':0.60,
           'merging_height_ratio_4':0.60,  'merging_gap_4':40,  'merging_percentage_4':0.70,
           'discarded_noise_selfAUC_1':11,    'discarded_noise_selfLength_1':5,   'discarded_noise_LoG_average_1':-0.06,
           'discarded_noise_selfLength_2':5,  'discarded_noise_LoG_average_2':-0.03,
           'discarded_noise_selfHeight_3':3,  'discarded_noise_LoG_average_3':-0.01,
           'discarded_noise_RealSelfLength_4':2,  'discarded_noise_LoG_average_4':-100.0,
           'discarded_noise_RealSelfLength_5':3,  'discarded_noise_LoG_average_5':-0.5,
           'discarded_noise_RealSelfLength_6':4,  'discarded_noise_LoG_average_6':-0.15}

########chromosome length in hg18
########chromosome_length={'chr1':247249719,'chr2':242951149,'chr3':199501827,'chr4':191273063,'chr5':180857866,'chr6':170899992,'chr7':158821424,'chr8':146274826,'chr9':140273252,'chr10':135374737,'chr11':134452384,'chr12':132349534,'chr13':114142980,'chr14':106368585,'chr15':100338915,'chr16':88827254,'chr17':78774742,'chr18':76117153,'chr19':63811651,'chr20':62435964,'chr21':46944323,'chr22':49691432,'chrX':154913754,'chrY':57772954}










### Main program
import math
import time
import sys
from optparse import OptionParser
import os
import pickle

def Gaussian_profile(PreliminaryConvolutionParameters):
    sigma=PreliminaryConvolutionParameters['sigma']
    times_of_sigma=PreliminaryConvolutionParameters['times_of_sigma']
    secondary_sigma=PreliminaryConvolutionParameters['secondary_sigma']
    Gaussian=[]
    First_Derivative_of_Gaussian=[]
    LoG=[]
    Third_Derivative_of_Gaussian=[]
    for x0 in range(-sigma*times_of_sigma,sigma*times_of_sigma+1):
        x=x0*(-1)
        Gaussian.append(math.exp(-x*x/(2.0*sigma*sigma))/math.sqrt(2*(math.pi)*sigma*sigma))
        First_Derivative_of_Gaussian.append(((-x)/((sigma*sigma)*(math.sqrt(2*(math.pi)*sigma*sigma))))*(math.exp(-x*x/(2.0*sigma*sigma))))
        LoG.append(((x*x)/(sigma*sigma*sigma*sigma)-1/(sigma*sigma))*(math.exp(-x*x/(2.0*sigma*sigma)))/(math.sqrt(2*(math.pi)*sigma*sigma)))
        Third_Derivative_of_Gaussian.append(((3*x)/(sigma*sigma*sigma*sigma)-(x*x*x)/(sigma*sigma*sigma*sigma*sigma*sigma))*(math.exp(-x*x/(2.0*sigma*sigma)))/(math.sqrt(2*(math.pi)*sigma*sigma)))
    secondary_Gaussian=[]
    secondary_First_Derivative_of_Gaussian=[]
    secondary_LoG=[]
    secondary_Third_Derivative_of_Gaussian=[]
    for x0 in range(-secondary_sigma*times_of_sigma,secondary_sigma*times_of_sigma+1):
        x=x0*(-1)
        secondary_Gaussian.append(math.exp(-x*x/(2.0*secondary_sigma*secondary_sigma))/math.sqrt(2*(math.pi)*secondary_sigma*secondary_sigma))
        secondary_First_Derivative_of_Gaussian.append(((-x)/((secondary_sigma*secondary_sigma)*(math.sqrt(2*(math.pi)*secondary_sigma*secondary_sigma))))*(math.exp(-x*x/(2.0*secondary_sigma*secondary_sigma))))
        secondary_LoG.append(((x*x)/(secondary_sigma*secondary_sigma*secondary_sigma*secondary_sigma)-1/(secondary_sigma*secondary_sigma))*(math.exp(-x*x/(2.0*secondary_sigma*secondary_sigma)))/(math.sqrt(2*(math.pi)*secondary_sigma*secondary_sigma)))
        secondary_Third_Derivative_of_Gaussian.append(((3*x)/(secondary_sigma*secondary_sigma*secondary_sigma*secondary_sigma)-(x*x*x)/(secondary_sigma*secondary_sigma*secondary_sigma*secondary_sigma*secondary_sigma*secondary_sigma))*(math.exp(-x*x/(2.0*secondary_sigma*secondary_sigma)))/(math.sqrt(2*(math.pi)*secondary_sigma*secondary_sigma)))
    ConvolutionParameters={'Gaussian':Gaussian,
                           'First_Derivative_of_Gaussian':First_Derivative_of_Gaussian,
                           'LoG':LoG,
                           'Third_Derivative_of_Gaussian':Third_Derivative_of_Gaussian,
                           'sigma':sigma,
                           'times_of_sigma':times_of_sigma,
                           'secondary_Gaussian':secondary_Gaussian,
                           'secondary_First_Derivative_of_Gaussian':secondary_First_Derivative_of_Gaussian,
                           'secondary_LoG':secondary_LoG,
                           'secondary_Third_Derivative_of_Gaussian':secondary_Third_Derivative_of_Gaussian,
                           'secondary_sigma':secondary_sigma}
    return ConvolutionParameters





def log10( x ):
    return math.log(x)/math.log(10)

class Poisson_test:
    def greater( background , X ):
        if X > 0:
            Added = math.exp( - background )
            #Added = 1
            for i in range( 1 , X ):
                Added = Added * background /i
            cdf = 0
            i = X
            while Added > 0:
                Added = Added * background / i
                cdf = cdf + Added
                i = i + 1
            #cdf = cdf * math.exp( - background )
            score = math.log(cdf)/math.log(10)
            score = round( -score , 8 )
        elif X == 0:
            cdf = 1
            score = 0
        return cdf , score

    def less( background , X ):
        Added = math.exp( - background )
        #Added = 1
        cdf = 0 + Added
        for i in range( 1 , X+1 ):
            Added = Added * background / i
            cdf = cdf + Added
        #cdf = cdf * math.exp( -1 background )
        score = math.log(cdf)/math.log(10)
        score = round( -score , 8 )
        return cdf , score

    def greater_fast( background , X ):
        if X > 0:
            using = 100
            resting = background - using
            Added = math.exp( - using )
            for i in range( 1 , X ):
                Added = Added * background /i
                if Added > 0:
                    while log10( Added ) > 200:
                        Added = Added * math.exp(-10)
                        resting = resting - 10
                    if log10( Added ) < 10:
                        Added = Added * math.exp(10)
                        resting = resting + 10
                else:
                    pass
            cdf = 0
            i = X
            while Added > 0:
                Added = Added * background / i
                cdf = cdf + Added
                i = i + 1
                if Added > 0:
                    while log10( Added ) > 200:
                        Added = Added * math.exp(-100)
                        cdf = cdf * math.exp(-100)
                        resting = resting - 100
                else:
                    pass
            score = log10( cdf )
            cdf = cdf * math.exp( - resting )
            score = score - resting / math.log(10)
            score = round( -score , 8 )
        elif X == 0:
            cdf = 1
            score = 0
        return cdf , score

    def less_fast( background , X ):
        using = 700
        resting = background - using
        Added = math.exp( - using )
        cdf = 0 + Added
        for i in range( 1 , X+1 ):
            Added = Added * background / i
            cdf = cdf + Added
            if Added > 0:
                while log10( Added ) > 200:
                    Added = Added * math.exp(-100)
                    cdf = cdf * math.exp(-100)
                    resting = resting - 100
            else:
                pass
        score = log10( cdf )
        cdf = cdf * math.exp( - resting )
        score = score - resting / math.log(10)
        score = round( -score , 8 )
        return cdf , score



class MainProgram:
    def __init__(self,FilesParameters,ConvolutionParameters,threshold):
        print(time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))
        self.inputfile=FilesParameters['inputfile_for_nucleosome_positioning']
        self.celltype=self.inputfile.split('/')[-1].split('Nucleosome')[0]
        self.chromosome=self.inputfile.split('/')[-1].split('.')[0].split('-')[-1]
        self.chrlength=chromosome_length[self.chromosome]
        self.chrRecordingListLength=0
        self.outputfile_path=FilesParameters['fileout_path']
        self.outputfile_wiggle=self.outputfile_path+self.inputfile.split('/')[-1][0]+'_'+self.chromosome+FilesParameters['outputfile_suffix']+'.like_wig'
        self.outputfile_nucleosome=self.outputfile_path+self.inputfile.split('/')[-1][0]+'_'+self.chromosome+FilesParameters['outputfile_suffix']+'.like_bed'
        print('Input file:  ',self.inputfile)
        print('Cell type:   ',self.celltype+' CD4+ T cells from human')
        print('Chromosome:  ',self.chromosome)
        print('Length of',self.chromosome,'in hg18:  ',self.chrlength)
        print('Output wiggle file of nucleosome distribution: ',self.outputfile_wiggle)
        print('Output file of nucleosome collection:          ',self.outputfile_nucleosome)
        self.ConvolutionParameters=ConvolutionParameters
        self.threshold=threshold
        self.score_list=[]
        self.tag_list = [ ]
        self.Gaussian_list=[]
        self.FDoG_list=[]
        self.LoG_list=[]
        self.TDoG_list=[]
        self.score_table=[]    ### coordinate, self.score_list, self.Gaussian_list, self.LoG_list, column for recording
        ##self.secondary_Gaussian_list=[]
        ##self.secondary_FDoG_list=[]
        self.secondary_LoG_list=[]
        ##self.secondary_TDoG_list=[]
        print('======> ======> ======> ======> ======> ======>')


    def score(self):
        print('Reading tag coodinate bed file and scoring the nucleosomes ......')
        if self.chrlength%10==0:
            self.chrRecordingListLength=self.chrlength//10
            self.score_list=[0]*(self.chrlength//10)
            self.tag_list = [0]*( self.chrlength//10 )
        elif self.chrlength%10!=0:
            self.chrRecordingListLength=self.chrlength//10+1
            self.score_list=[0]*(self.chrlength//10+1)
            self.tag_list = [0]*( self.chrlength//10+1 )
        line=0
        text=open(self.inputfile,'r')
        while True:
            try:
                data=next(text).split('\n')[0].split('\r')[0].split('\t')
                line=line+1
                # sys.stdout.write('\r        Line:'+str(line) )
                if data[5]=='+':
                    beginning=int(data[1])+37
                    ending=int(data[1])+37+75
                    TagCoordinate = int( data[1] ) + 75
                elif data[5]=='-':
                    beginning=int(data[2])-37-75
                    ending=int(data[2])-37
                    TagCoordinate = int( data[2] ) - 75
                ####
                for i in range(beginning,ending+1):
                    if ( ((i-1)//10)<=self.chrRecordingListLength-1 ) and ( ((i-1)//10)>=0 ):
                        self.score_list[(i-1)//10]+=0.1
                    else:
                        ####print('Coordinate error in ',self.inputfile,'line:',line,' ',data, i, self.chrRecordingListLength-1)      #### #### #### #### 20141021 #### #### #### ####
                        pass                                                                                                          #### #### #### #### 20141021 #### #### #### ####
                ####
                if ( ((TagCoordinate-1)//10) <= self.chrRecordingListLength - 1 ) and ( ((TagCoordinate-1)//10)>=0 ):
                    self.tag_list[ (TagCoordinate-1)//10 ] += 1
                else:
                    ####print('Coordinate error in ',self.inputfile,'line:',line,' ',data, i, self.chrRecordingListLength-1)
                    pass
                ####
                # sys.stdout.flush()
            except StopIteration:
                break
        text.close()
        print('...... File reading and nucleosome scoring is finished.','\t',time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))
        print('\tChecking ==> Total tag fragments:  ',line)
        print('\tChecking ==> Length of original self.score_list:  ',len(self.score_list))
        print('\tChecking ==> Length of original self.tag_list:    ',len(self.tag_list) )


    def score_for_Paired_end(self):
        print('Read paired-end tags and scoring the nucleosomes ......')
        if self.chrlength%10==0:
            self.chrRecordingListLength=self.chrlength//10
            self.score_list=[0]*(self.chrlength//10)
            self.tag_list = [0]*( self.chrlength//10 )
        elif self.chrlength%10!=0:
            self.chrRecordingListLength=self.chrlength//10+1
            self.score_list=[0]*(self.chrlength//10+1)
            self.tag_list = [0]*( self.chrlength//10+1 )
        line=0
        text=open(self.inputfile,'r')
        while True:
            try:
                data=next(text).split('\n')[0].split('\r')[0].split('\t')
                line=line+1
                # sys.stdout.write('\r        Line:'+str(line) )
                #if data[5]=='+':
                #    beginning=int(data[1])+37
                #    ending=int(data[1])+37+75
                #    TagCoordinate = int( data[1] ) + 75
                #elif data[5]=='-':
                #    beginning=int(data[2])-37-75
                #    ending=int(data[2])-37
                #    TagCoordinate = int( data[2] ) - 75
                A = int(data[1])
                B = int(data[2])
                tag_length = B - A + 1
                if ( tag_length >= FilesParameters[ 'pe_min' ] ) and ( tag_length <= FilesParameters[ 'pe_max' ] ):
                    beginning = round( A + 0.25*(B-A) )
                    ending    = round( B - 0.25*(B-A) )
                    TagCoordinate = round( (A+B)*0.5 )
                    ####
                    for i in range(beginning,ending+1):
                        if ( ((i-1)//10)<=self.chrRecordingListLength-1 ) and ( ((i-1)//10)>=0 ):
                            self.score_list[(i-1)//10]+=0.1
                        else:
                            ####print('Coordinate error in ',self.inputfile,'line:',line,' ',data, i, self.chrRecordingListLength-1)      #### #### #### #### 20141021 #### #### #### ####
                            pass                                                                                                          #### #### #### #### 20141021 #### #### #### ####
                    ####
                    if ( ((TagCoordinate-1)//10) <= self.chrRecordingListLength - 1 ) and ( ((TagCoordinate-1)//10)>=0 ):
                        self.tag_list[ (TagCoordinate-1)//10 ] += 1
                    else:
                        ####print('Coordinate error in ',self.inputfile,'line:',line,' ',data, i, self.chrRecordingListLength-1)
                        pass
                    ####
                else:
                    pass
                # sys.stdout.flush()
            except StopIteration:
                break
        text.close()
        print('...... File reading and nucleosome scoring is finished.','\t',time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))
        print('\tChecking ==> Total tag fragments:  ',line)
        print('\tChecking ==> Length of original self.score_list:  ',len(self.score_list))
        print('\tChecking ==> Length of original self.tag_list:    ',len(self.tag_list) )
        


    def Gaussian_convolution_smoothing(self):
        print('Performing Gaussian convolution ......')
        def convolution(x,Gaussian,First_Derivative_of_Gaussian,LoG,Third_Derivative_of_Gaussian,sigma,times_of_sigma):    ### x is the index number of self.score_list, which is transmitted by parameter j in the following function "Gaussian_convolution_smoothing".
            y=0
            FDoG_y=0
            LoG_y=0
            TDoG_y=0
            if x>=sigma*times_of_sigma and x<=len(self.score_list)-sigma*times_of_sigma-1:
                for n in range(x-sigma*times_of_sigma,x+sigma*times_of_sigma+1):
                    y=y+self.score_list[n]*Gaussian[n-(x-sigma*times_of_sigma)]
                    FDoG_y=FDoG_y+self.score_list[n]*First_Derivative_of_Gaussian[n-(x-sigma*times_of_sigma)]
                    LoG_y=LoG_y+self.score_list[n]*LoG[n-(x-sigma*times_of_sigma)]
                    TDoG_y=TDoG_y+self.score_list[n]*Third_Derivative_of_Gaussian[n-(x-sigma*times_of_sigma)]
            elif x<sigma*times_of_sigma:
                for n in range(0,x+sigma*times_of_sigma+1):
                    y=y+self.score_list[n]*Gaussian[sigma*times_of_sigma-x+n]
                    FDoG_y=FDoG_y+self.score_list[n]*First_Derivative_of_Gaussian[sigma*times_of_sigma-x+n]
                    LoG_y=LoG_y+self.score_list[n]*LoG[sigma*times_of_sigma-x+n]
                    TDoG_y=TDoG_y+self.score_list[n]*Third_Derivative_of_Gaussian[sigma*times_of_sigma-x+n]
            elif x>len(self.score_list)-sigma*times_of_sigma-1:
                for n in range(x-sigma*times_of_sigma,len(self.score_list)):
                    y=y+self.score_list[n]*Gaussian[sigma*times_of_sigma-x+n]
                    FDoG_y=FDoG_y+self.score_list[n]*First_Derivative_of_Gaussian[sigma*times_of_sigma-x+n]
                    LoG_y=LoG_y+self.score_list[n]*LoG[sigma*times_of_sigma-x+n]
                    TDoG_y=TDoG_y+self.score_list[n]*Third_Derivative_of_Gaussian[sigma*times_of_sigma-x+n]
            return y,FDoG_y,LoG_y,TDoG_y

        def convolution_secondary(x,Gaussian,First_Derivative_of_Gaussian,LoG,Third_Derivative_of_Gaussian,sigma,times_of_sigma):    ### x is the index number of self.score_list, which is transmitted by parameter j in the following function "Gaussian_convolution_smoothing".
            #y=0
            #FDoG_y=0
            LoG_y=0
            #TDoG_y=0
            if x>=sigma*times_of_sigma and x<=len(self.score_list)-sigma*times_of_sigma-1:
                for n in range(x-sigma*times_of_sigma,x+sigma*times_of_sigma+1):
                    #y=y+self.score_list[n]*Gaussian[n-(x-sigma*times_of_sigma)]
                    #FDoG_y=FDoG_y+self.score_list[n]*First_Derivative_of_Gaussian[n-(x-sigma*times_of_sigma)]
                    LoG_y=LoG_y+self.score_list[n]*LoG[n-(x-sigma*times_of_sigma)]
                    #TDoG_y=TDoG_y+self.score_list[n]*Third_Derivative_of_Gaussian[n-(x-sigma*times_of_sigma)]
            elif x<sigma*times_of_sigma:
                for n in range(0,x+sigma*times_of_sigma+1):
                    #y=y+self.score_list[n]*Gaussian[sigma*times_of_sigma-x+n]
                    #FDoG_y=FDoG_y+self.score_list[n]*First_Derivative_of_Gaussian[sigma*times_of_sigma-x+n]
                    LoG_y=LoG_y+self.score_list[n]*LoG[sigma*times_of_sigma-x+n]
                    #TDoG_y=TDoG_y+self.score_list[n]*Third_Derivative_of_Gaussian[sigma*times_of_sigma-x+n]
            elif x>len(self.score_list)-sigma*times_of_sigma-1:
                for n in range(x-sigma*times_of_sigma,len(self.score_list)):
                    #y=y+self.score_list[n]*Gaussian[sigma*times_of_sigma-x+n]
                    #FDoG_y=FDoG_y+self.score_list[n]*First_Derivative_of_Gaussian[sigma*times_of_sigma-x+n]
                    LoG_y=LoG_y+self.score_list[n]*LoG[sigma*times_of_sigma-x+n]
                    #TDoG_y=TDoG_y+self.score_list[n]*Third_Derivative_of_Gaussian[sigma*times_of_sigma-x+n]
            return LoG_y  ##y,FDoG_y,LoG_y,TDoG_y

        ########
        Gaussian=self.ConvolutionParameters['Gaussian']
        First_Derivative_of_Gaussian=self.ConvolutionParameters['First_Derivative_of_Gaussian']
        LoG=self.ConvolutionParameters['LoG']
        Third_Derivative_of_Gaussian=self.ConvolutionParameters['Third_Derivative_of_Gaussian']
        sigma=self.ConvolutionParameters['sigma']
        ########
        times_of_sigma=self.ConvolutionParameters['times_of_sigma']
        ########
        secondary_Gaussian=self.ConvolutionParameters['secondary_Gaussian']
        secondary_First_Derivative_of_Gaussian=self.ConvolutionParameters['secondary_First_Derivative_of_Gaussian']
        secondary_LoG=self.ConvolutionParameters['secondary_LoG']
        secondary_Third_Derivative_of_Gaussian=self.ConvolutionParameters['secondary_Third_Derivative_of_Gaussian']
        secondary_sigma=self.ConvolutionParameters['secondary_sigma']
        ########
        for j in range(len(self.score_list)):
            # sys.stdout.write('\r        Coordinate:'+str(j*10+1) )
            Gaussian_result,FDoG_result,LoG_result,TDoG_result=convolution(j,Gaussian,First_Derivative_of_Gaussian,LoG,Third_Derivative_of_Gaussian,sigma,times_of_sigma)
            self.Gaussian_list.append(Gaussian_result)
            self.FDoG_list.append(FDoG_result)
            self.LoG_list.append(LoG_result)
            self.TDoG_list.append(TDoG_result)
            ########
            self.score_table.append([j*10+1,self.score_list[j],Gaussian_result,LoG_result,0])
            ########
            ##secondary_Gaussian_result,secondary_FDoG_result,secondary_LoG_result,secondary_TDoG_result=convolution(j,secondary_Gaussian,secondary_First_Derivative_of_Gaussian,secondary_LoG,secondary_Third_Derivative_of_Gaussian,secondary_sigma,times_of_sigma)
            secondary_LoG_result = convolution_secondary(j,secondary_Gaussian,secondary_First_Derivative_of_Gaussian,secondary_LoG,secondary_Third_Derivative_of_Gaussian,secondary_sigma,times_of_sigma)
            ##self.secondary_Gaussian_list.append(secondary_Gaussian_result)
            ##self.secondary_FDoG_list.append(secondary_FDoG_result)
            self.secondary_LoG_list.append(secondary_LoG_result)
            ##self.secondary_TDoG_list.append(secondary_TDoG_result)
            # sys.stdout.flush()
        print('...... Convolution is finished.','\t',time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))
        print('\tChecking ==> Length of self.Gaussian_list:  ',len(self.Gaussian_list))##,'\tSecondary Length of self.Gaussian_list:  ',len(self.secondary_Gaussian_list))
        print('\tChecking ==> Length of self.FDoG_list:  ',len(self.FDoG_list))##,'\tSecondary Length of self.FDoG_list:  ',len(self.secondary_FDoG_list))
        print('\tChecking ==> Length of self.LoG_list:   ',len(self.LoG_list),'\tSecondary Length of self.LoG_list:   ',len(self.secondary_LoG_list))
        print('\tChecking ==> Length of self.TDoG_list:  ',len(self.TDoG_list))##,'\tSecondary Length of self.TDoG_list:  ',len(self.secondary_TDoG_list))


    def extremum_detection(self):
        print('Detecting extremum for peaks and valleys ......')
        ######## 重要记录器 ###################################
        self.peak_dict={}
        self.peak_details_dict={}
        ######## 重要记录器 ###################################
        one_peak=['left_valley_ending_coordinate','max_left_beginning_coordinate','max_right_ending_coordinate','right_valley_beginning_coordinate',
                  'left_valley_ending_index','max_left_beginning_index','max_right_ending_index','right_valley_beginning_index',
                  'serial_number']
        one_peak_finished='none'
        holding_on=0    ### record plarform
        above_or_below=0    ### record going up or going down: 0/1/-1
        serial_number=0
        for t in range(1,len(self.score_table)):
            if self.FDoG_list[t]>0:    ### --> 上坡
                if above_or_below==0:    ### 记录第一个peak的起始
                    one_peak[0]=self.score_table[t][0]
                    one_peak[4]=t
                elif above_or_below==1:    ### 如果原本即为上坡，把误认为峰顶的平台取消
                    one_peak[1]='max_left_beginning_coordinate'
                    one_peak[5]='max_left_beginning_index'
                elif above_or_below==-1:    ### 如果原本为下坡，记录一个新的peak的起始
                    if type(one_peak[3])!=int or type(one_peak[7])!=int:
                        one_peak[3]=self.score_table[t-1][0]
                        one_peak[7]=t-1
                    one_peak_finished=one_peak[:]
                    one_peak=['left_valley_ending_coordinate','max_left_beginning_coordinate','max_right_ending_coordinate','right_valley_beginning_coordinate','left_valley_ending_index','max_left_beginning_index','max_right_ending_index','right_valley_beginning_index','serial_number']
                    one_peak[0]=self.score_table[t][0]
                    one_peak[4]=t
                above_or_below=1
                holding_on=0
            elif self.FDoG_list[t]<0:    ### --> 下坡
                if above_or_below==-1:
                    one_peak[3]='right_valley_beginning_coordinate'
                    one_peak[7]='right_valley_beginning_index'
                elif above_or_below==1:
                    one_peak[2]=self.score_table[t-holding_on][0]
                    one_peak[6]=t-holding_on
                    if type(one_peak[1])!=int or type(one_peak[5])!=int:
                        one_peak[1]=self.score_table[t-max(1,holding_on)][0]
                        one_peak[5]=t-max(1,holding_on)
                above_or_below=-1
                holding_on=0
            elif self.FDoG_list[t]==0:    ### --> 平台
                holding_on=holding_on+1
                if above_or_below==1 and self.FDoG_list[t-1]>0:
                    one_peak[1]=self.score_table[t][0]
                    one_peak[5]=t
                elif above_or_below==-1 and self.FDoG_list[t-1]<0:
                    one_peak[3]=self.score_table[t-1][0]
                    one_peak[7]=t-1
            if t==len(self.score_table)-1:
                if self.FDoG_list[t]<0:                                       ######## 2014-10-23-pm-23-22 ########
                    if above_or_below==-1:                                    ######## 2014-10-23-pm-23-22 ########
                        if type(one_peak[3])!=int or type(one_peak[7])!=int:  ######## 2014-10-23-pm-23-22 ########
                            one_peak[3]=self.score_table[t][0]                ######## 2014-10-23-pm-23-22 ########
                            one_peak[7]=t                                     ######## 2014-10-23-pm-23-22 ########
                one_peak_finished=one_peak[:]
            if one_peak_finished!='none' and type(one_peak_finished)==list and type(one_peak_finished[0])==int and type(one_peak_finished[1])==int and type(one_peak_finished[2])==int and type(one_peak_finished[3])==int and type(one_peak_finished[4])==int and type(one_peak_finished[5])==int and type(one_peak_finished[6])==int and type(one_peak_finished[7])==int:
                serial_number=serial_number+1
                one_peak_finished[8]=serial_number
                self.peak_dict[serial_number]=one_peak_finished    ### 记录
                for p in range(one_peak_finished[4],one_peak_finished[7]+1):
                    self.peak_details_dict[p]=one_peak_finished    ### 记录
                one_peak_finished='none'
        print('\tChecking ==> Maximum peak serial number:  ',serial_number)
        print('...... Extremum detection for peaks and valleys is finished.','\t',time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))
        print('\tChecking ==> The number of peaks:  ',len(self.peak_dict))


    def inflection_pairs_detection(self):
        print('Detecting inflection pairs for nucleosome candidates ......')
        ######## 重要记录器 ###################################
        self.inflection_pairs_list=[]
        ######## 重要记录器 ###################################
        one_pair_of_inflection=['unknown','unknown','n1','n2','nindex']
        holding_on=0
        above_or_below=0
        nindex=0
        for t in range(1,len(self.score_table)):
            if self.score_table[t][3]>0:
                holding_on=0
                if above_or_below==1:
                    pass
                elif above_or_below!=1:
                    if type(one_pair_of_inflection[0])==int and type(one_pair_of_inflection[2])==int:
                        one_pair_of_inflection[1]=self.score_table[t-1][0]
                        one_pair_of_inflection[3]=t-1
                    else:
                        pass
                    above_or_below=1
            elif self.score_table[t][3]<0:
                holding_on=0
                if above_or_below==-1:
                    pass
                elif above_or_below!=-1:
                    one_pair_of_inflection[0]=self.score_table[t][0]
                    one_pair_of_inflection[2]=t
                    one_pair_of_inflection[1]='unknown'
                    one_pair_of_inflection[3]='n2'
                    above_or_below=-1
            elif self.score_table[t][3]==0:
                holding_on=holding_on+1
            ### Test the current "one_pair_of_inflection" list.
            if type(one_pair_of_inflection[0])==int and type(one_pair_of_inflection[1])==int and type(one_pair_of_inflection[2])==int and type(one_pair_of_inflection[3])==int:
                nindex=nindex+1
                one_pair_of_inflection[4]=nindex
                self.inflection_pairs_list.append(one_pair_of_inflection)
                one_pair_of_inflection=['unknown','unknown','n1','n2','nindex']
        print('\tChecking ==> Maximun inflection pair index number:  ',nindex)
        print('...... Inflection pairs collection is finished.','\t',time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))
        print('\tChecking ==> Total inflection pairs for nucleosome candidates:  ',len(self.inflection_pairs_list))


    def inflection_pairs_midpoint_detection(self):
        print('Detecting midpoint of every inflection pairs ......')
        midpoints_candidates_list=[]    ### Recording
        midpoints_candidates_dict={}    ### Recording
        midpoints_candidates_details_dict={}    ### Recording
        midpoints_candidates_index=0
        one_midpoint=['unknown','unknown','m1','m2','mindex']
        holding_on=0
        above_or_below=0
        for t in range(1,len(self.score_table)):
            if self.TDoG_list[t]>0:
                if above_or_below==1:
                    pass
                elif above_or_below==-1:
                    one_midpoint[1]=self.score_table[t][0]
                    one_midpoint[3]=t
                    one_midpoint[0]=self.score_table[t-max(1,holding_on)][0]
                    one_midpoint[2]=t-max(1,holding_on)
                above_or_below=1
                holding_on=0
            elif self.TDoG_list[t]<0:
                above_or_below=-1
                holding_on=0
            elif self.TDoG_list[t]==0:
                holding_on=holding_on+1
            if type(one_midpoint[0])==int and type(one_midpoint[1])==int and type(one_midpoint[2])==int and type(one_midpoint[3])==int:
                midpoints_candidates_index=midpoints_candidates_index+1
                one_midpoint[4]=midpoints_candidates_index
                midpoints_candidates_list.append(one_midpoint)    ### Recording
                midpoints_candidates_dict[midpoints_candidates_index]=one_midpoint   ### Recording
                for m in range(one_midpoint[2],one_midpoint[3]+1):
                    midpoints_candidates_details_dict[m]=one_midpoint    ### Recording
                one_midpoint=['unknown','unknown','m1','m2','mindex']
        print('...... The detection for all midpoint candidates of every inflection pairs is finished.','\t',time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))
        ######## 重要记录器 ###################################
        self.inflection_pairs_list_with_midpoint=[]
        self.inflection_pairs_dict_with_midpoint={}
        ######## 重要记录器 ###################################
        real_midpoints_index=0
        for a_pair_of_inflection in self.inflection_pairs_list:
            a_pair_of_inflection_with_midpoint=a_pair_of_inflection[:]
            mapped_midpoint_mindex=[]
            for point in range(a_pair_of_inflection[2],a_pair_of_inflection[3]+1):
                if point in midpoints_candidates_details_dict.keys():
                    if midpoints_candidates_details_dict[point][4] not in mapped_midpoint_mindex:
                        mapped_midpoint_mindex.append(midpoints_candidates_details_dict[point][4])
                    else:
                        pass
            if len(mapped_midpoint_mindex)>0:
                real_midpoints_index=real_midpoints_index+len(mapped_midpoint_mindex)
                for pointt in mapped_midpoint_mindex:
                    a_pair_of_inflection_with_midpoint.append(midpoints_candidates_dict[pointt])    ### 给一对拐点注明其“中点”
            else:
                pass
            self.inflection_pairs_list_with_midpoint.append(a_pair_of_inflection_with_midpoint)
            self.inflection_pairs_dict_with_midpoint[a_pair_of_inflection_with_midpoint[4]]=a_pair_of_inflection_with_midpoint
        print('\tChecking ==> Maximun midpoint candidate index number:  ',midpoints_candidates_index)
        print('\tChecking ==> Maximun real midpoints index number:  ',real_midpoints_index)
        print('...... All midpoint candidates are integrated with inflection pairs.','\t',time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))
        print('\tChecking ==> Total inflection pairs with midpoint:  ',len(self.inflection_pairs_list_with_midpoint))


    def secondary_inflection_pairs_detection(self):
        print('Detecting secondary inflection pairs for nucleosome candidates border adjustment ......')
        ######## 重要记录器 ###################################
        self.secondary_inflection_pairs_list=[]
        ######## 重要记录器 ###################################
        one_pair_of_inflection=['unknown','unknown','n1','n2','nindex']
        holding_on=0
        above_or_below=0
        snindex=0
        for t in range(1,len(self.score_table)):
            if self.secondary_LoG_list[t]>0:
                holding_on=0
                if above_or_below==1:
                    pass
                elif above_or_below!=1:
                    if type(one_pair_of_inflection[0])==int and type(one_pair_of_inflection[2])==int:
                        one_pair_of_inflection[1]=self.score_table[t-1][0]
                        one_pair_of_inflection[3]=t-1
                    else:
                        pass
                    above_or_below=1
            elif self.secondary_LoG_list[t]<0:
                holding_on=0
                if above_or_below==-1:
                    pass
                elif above_or_below!=-1:
                    one_pair_of_inflection[0]=self.score_table[t][0]
                    one_pair_of_inflection[2]=t
                    one_pair_of_inflection[1]='unknown'
                    one_pair_of_inflection[3]='n2'
                    above_or_below=-1
            elif self.secondary_LoG_list[t]==0:
                holding_on=holding_on+1
            ### Test the current "one_pair_of_inflection" list.
            if type(one_pair_of_inflection[0])==int and type(one_pair_of_inflection[1])==int and type(one_pair_of_inflection[2])==int and type(one_pair_of_inflection[3])==int:
                snindex=snindex+1
                one_pair_of_inflection[4]=snindex
                self.secondary_inflection_pairs_list.append(one_pair_of_inflection)
                one_pair_of_inflection=['unknown','unknown','n1','n2','nindex']
            else:
                pass
        print('\tChecking ==> Total secondary inflection pairs:  ',snindex)
        ### 直接与peak的信息整合成为字典
        ######## 重要记录器 ###################################
        self.secondary_inflection_pairs_dict_with_peak={}
        ######## 重要记录器 ###################################
        snindex_integrated_with_peak=0
        for one_pair_of_inflection in self.secondary_inflection_pairs_list:
            peak_index_list=[]    ### 记录主峰序号
            for n in range(one_pair_of_inflection[2],one_pair_of_inflection[3]+1):
                if n in self.peak_details_dict.keys():
                    if self.peak_details_dict[n][8] not in peak_index_list:
                        peak_index_list.append(self.peak_details_dict[n][8])
                    else:
                        pass
            for peak_index in peak_index_list:
                if peak_index not in self.secondary_inflection_pairs_dict_with_peak.keys():
                    self.secondary_inflection_pairs_dict_with_peak[peak_index]=[one_pair_of_inflection]
                    snindex_integrated_with_peak=snindex_integrated_with_peak+1
                elif peak_index in self.secondary_inflection_pairs_dict_with_peak.keys():
                    self.secondary_inflection_pairs_dict_with_peak[peak_index].append(one_pair_of_inflection)
                    snindex_integrated_with_peak=snindex_integrated_with_peak+1
        print('\tChecking ==> Total secondary inflection pairs integrated with peak:  ',snindex_integrated_with_peak)
        print('...... Secondary inflection pairs collection is finished.','\t',time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))
            

    def preliminary_nucleosome_position(self):
        print('Beginning preliminary main nucleosome and shoulder positioning ......')
        integrated_pairs_counting=0    ### 计数被整合入peak_dict的inflection pairs
        ######## 重要记录器 ###################################
        self.unintegrated_inflection_pairs_list=[]
        self.nucleosome_list=[]
        self.nucleosome_dict={}
        self.shoulder_list=[]
        self.shoulder_dict={}
        ######## 重要记录器 ###################################
        for one_pair_of_inflection in self.inflection_pairs_list_with_midpoint:
            ### Check the middle peak between two inflection points, and the two valley outside of the two inflection points.
            peak_checking=0    ### checking point 1
            valley_left_checking=0    ### checking point 2
            valley_right_checking=0    ### checking point 3
            ### Prepare a new list to collect the max-extremun between a pair of inflection
            peak_index_list=[]
            ### To find the max-extremun between the pair of inflection
            for n in range(one_pair_of_inflection[2],one_pair_of_inflection[3]+1):
                if n in self.peak_details_dict.keys():
                    if self.peak_details_dict[n][8] not in peak_index_list:
                        peak_index_list.append(self.peak_details_dict[n][8])
                    else:
                        pass
            ### There should be only one max-extremun between the pair of inflection.
            ### If there is only one max-extremun between the pair of inflection, check the peak-max-extremun-point, and the beginning and ending points of the peak.
            if len(peak_index_list)==1:
                ### 记录主峰序号
                peak_index=peak_index_list[0]
                    
                ### 在self.peak_dict中记录该inflection pair
                self.peak_dict[peak_index].append(one_pair_of_inflection)    ### 此时对self.peak_dict进行更新，此后包含inflection pairs
                integrated_pairs_counting=integrated_pairs_counting+1
                    
                ### peak的8个位置参数
                max_beginning=self.peak_dict[peak_index][5]
                max_beginning_coordinate=self.peak_dict[peak_index][1]
                max_ending=self.peak_dict[peak_index][6]
                max_ending_coordinate=self.peak_dict[peak_index][2]
                peak_beginning=self.peak_dict[peak_index][4]
                peak_beginning_coordinate=self.peak_dict[peak_index][0]
                peak_ending=self.peak_dict[peak_index][7]
                peak_ending_coordinate=self.peak_dict[peak_index][3]
                ### 鉴定该inflection pair的极值和边界
                if (self.peak_dict[peak_index][5]>=one_pair_of_inflection[2] and self.peak_dict[peak_index][5]<=one_pair_of_inflection[3]) or (self.peak_dict[peak_index][6]>=one_pair_of_inflection[2] and self.peak_dict[peak_index][6]<=one_pair_of_inflection[3]):    ####peak_dict[peak_index][5]>=one_pair_of_inflection[2] and peak_dict[peak_index][6]<=one_pair_of_inflection[3]:
                    peak_checking=1    ### checking point 1 OK
                if one_pair_of_inflection[2]>=self.peak_dict[peak_index][4]:
                    valley_left_checking=1    ### checking point 2 OK
                if one_pair_of_inflection[3]<=self.peak_dict[peak_index][7]:
                    valley_right_checking=1    ### checking point 3 OK
            else:
                ### 在unintegrated_inflection_pairs_list中记录无用的inflection pair
                self.unintegrated_inflection_pairs_list.append(one_pair_of_inflection)
                
            ### If it is a main nucleosome, make notes in self.score_table.
            ### If it is a shoulder, collect it in shoulder list.
            if peak_checking==1 and valley_left_checking==1 and valley_right_checking==1:    ###(This is a main nucleosome.)
                ### 记录一个nucleosome的起讫，和左右的两个谷，以及所在peak的编号。
                ### 以列表和字典（以所在peak的编号为key）两种格式记录。
                ### Notes: BC -- beginning_coordinate, EC -- ending_coordinate, BI -- beginning_index, EI -- ending_index
                ### self.nucleosome_list=[           [pairBC, pairEC, pairBI, pairEI, pair_index, 'height', 'area', [peakBC, maxBC, maxEC, peakEC, peakBI, maxBI, maxEI, peakEI, peak_index]], ...]
                ### self.nucleosome_dict={peak_index:[pairBC, pairEC, pairBI, pairEI, pair_index, 'height', 'area', [peakBC, maxBC, maxEC, peakEC, peakBI, maxBI, maxEI, peakEI, peak_index]], ...}
                self.nucleosome_list.append([one_pair_of_inflection[0],one_pair_of_inflection[1],one_pair_of_inflection[2],one_pair_of_inflection[3],one_pair_of_inflection[4],'max(original_score_fragment)','area',[peak_beginning_coordinate,max_beginning_coordinate,max_ending_coordinate,peak_ending_coordinate,peak_beginning,max_beginning,max_ending,peak_ending,peak_index]])
                self.nucleosome_dict[peak_index]=[one_pair_of_inflection[0],one_pair_of_inflection[1],one_pair_of_inflection[2],one_pair_of_inflection[3],one_pair_of_inflection[4],'max(original_score_fragment)','area',[peak_beginning_coordinate,max_beginning_coordinate,max_ending_coordinate,peak_ending_coordinate,peak_beginning,max_beginning,max_ending,peak_ending,peak_index]]
            elif peak_checking==0 and valley_left_checking==1 and valley_right_checking==1:    ###(This is a shoulder.)
                ### self.shoulder_list=  [           [pairBC, pairEC, pairBI, pairEI, pair_index, 'height', 'area', [peakBC, maxBC, maxEC, peakEC, peakBI, maxBI, maxEI, peakEI, peak_index]], ...]
                ### self.shoulder_dict=  {pair_index:[pairBC, pairEC, pairBI, pairEI, pair_index, 'height', 'area', [peakBC, maxBC, maxEC, peakEC, peakBI, maxBI, maxEI, peakEI, peak_index]], ...}
                self.shoulder_list.append([one_pair_of_inflection[0],one_pair_of_inflection[1],one_pair_of_inflection[2],one_pair_of_inflection[3],one_pair_of_inflection[4],'max(original_score_fragment)','area',[peak_beginning_coordinate,max_beginning_coordinate,max_ending_coordinate,peak_ending_coordinate,peak_beginning,max_beginning,max_ending,peak_ending,peak_index]])
                self.shoulder_dict[one_pair_of_inflection[4]]=[one_pair_of_inflection[0],one_pair_of_inflection[1],one_pair_of_inflection[2],one_pair_of_inflection[3],one_pair_of_inflection[4],'max(original_score_fragment)','area',[peak_beginning_coordinate,max_beginning_coordinate,max_ending_coordinate,peak_ending_coordinate,peak_beginning,max_beginning,max_ending,peak_ending,peak_index]]
            else:
                pass
        print('...... Preliminary main nucleosome positioning is finished.','\t',time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))
        print('\tChecking ==> Total number of inflection pairs integrated into the peak dictionary:  ',integrated_pairs_counting)
        print('\tChecking ==> Excluded inflection pairs:  ',len(self.unintegrated_inflection_pairs_list))
        print('\tChecking ==> Total main nucleosome:  ',len(self.nucleosome_list))
        print('\tChecking ==> Number of shoulders:  ',len(self.shoulder_list))



    def collect_and_sort_shoulders(self):
        print('Collecting and sorting the shoulders with main nucleosomes in every peak region ......')
        ### Notes: BC -- beginning_coordinate, EC -- ending_coordinate, BI -- beginning_index, EI -- ending_index
        ### self.peak_dict={peak_index:[peakBC, maxBC, maxEC, peakEC, peakBI, maxBI, maxEI, peakEI, peak_index, [inflection pair, [midpoint], ...], ... ], ...}
        peak_nucleosome_shoulder_simple_collection={}
        for one_shoulder in self.shoulder_list:
            inflection_pair_index=one_shoulder[4]    ### 这个shoulder对应的inflection pair编号
            peak_index=one_shoulder[-1][-1]    ### 这个shoulder所处的peak的编号
            if peak_index in self.nucleosome_dict.keys():
                if peak_index not in peak_nucleosome_shoulder_simple_collection.keys():
                    peak_nucleosome_shoulder_simple_collection[peak_index]=[]
                    peak_nucleosome_shoulder_simple_collection[peak_index].append(inflection_pair_index)
                elif peak_index in peak_nucleosome_shoulder_simple_collection.keys():
                    peak_nucleosome_shoulder_simple_collection[peak_index].append(inflection_pair_index)
            elif peak_index not in self.nucleosome_dict.keys():
                pass

        ### 整理peak_nucleosome_shoulder_simple_collection并按shoulder出现的先后位置排序。
        ######## 重要记录器 ###################################
        self.shoulders_sorted_with_main_nucleosomes={}
        ######## 重要记录器 ###################################
        for peak_index in sorted( peak_nucleosome_shoulder_simple_collection.keys() ):
            self.shoulders_sorted_with_main_nucleosomes[peak_index]={}    ### 准备记录
            main_nucleosome_in_the_peak=self.nucleosome_dict[peak_index]
            left_part=[]
            right_part=[]
            for inflection_pair_index in peak_nucleosome_shoulder_simple_collection[peak_index]:
                if self.inflection_pairs_dict_with_midpoint[inflection_pair_index][0]>self.nucleosome_dict[peak_index][1]:
                    right_part.append(inflection_pair_index)
                elif self.inflection_pairs_dict_with_midpoint[inflection_pair_index][1]<self.nucleosome_dict[peak_index][0]:
                    left_part.append(inflection_pair_index)
            if len(left_part)==1:
                self.shoulders_sorted_with_main_nucleosomes[peak_index][-1]=left_part[0]
            elif len(left_part)>1:
                ### 按从大到小排序
                for i in range(0,len(left_part)-1):
                    for j in range(i+1,len(left_part)):
                        if self.inflection_pairs_dict_with_midpoint[left_part[i]][1]<self.inflection_pairs_dict_with_midpoint[left_part[j]][0]:
                            temp=left_part[i]
                            left_part[i]=left_part[j]
                            left_part[j]=temp
                        else:
                            pass
                for k in range(len(left_part)):
                    self.shoulders_sorted_with_main_nucleosomes[peak_index][(k+1)*(-1)]=left_part[k]
            if len(right_part)==1:
                self.shoulders_sorted_with_main_nucleosomes[peak_index][1]=right_part[0]
            elif len(right_part)>1:
                ### 按从小到大排序
                for i in range(0,len(right_part)-1):
                    for j in range(i+1,len(right_part)):
                        if self.inflection_pairs_dict_with_midpoint[right_part[i]][0]>self.inflection_pairs_dict_with_midpoint[right_part[j]][1]:
                            temp=right_part[i]
                            right_part[i]=right_part[j]
                            right_part[j]=temp
                        else:
                            pass
                for k in range(len(right_part)):
                    self.shoulders_sorted_with_main_nucleosomes[peak_index][(k+1)]=right_part[k]
        print('...... Shoulders are collected and sorted with main nucleosomes in every peak region.','\t',time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))
        


    def precise_nucleosome_position(self):
        print('Beginning precise nucleosome positioning ......')
        def Pearson_Correlation_Coefficient(Original,Smoothed):
            meanO=sum(Original)/len(Original)
            meanS=sum(Smoothed)/len(Smoothed)
            sumOO=0
            sumSS=0
            sumOS=0
            for i in range(len(Original)):
                sumOO=sumOO+(Original[i]-meanO)*(Original[i]-meanO)
                sumSS=sumSS+(Smoothed[i]-meanS)*(Smoothed[i]-meanS)
                sumOS=sumOS+(Original[i]-meanO)*(Smoothed[i]-meanS)
            pcc_result=(sumOS+0.00000001)/math.sqrt((sumOO+0.00000001)*(sumSS+0.00000001))
            return pcc_result

        def concavity_and_convexity(Smoothed):
            background=min(Smoothed)
            k=(Smoothed[-1]-Smoothed[0])/len(Smoothed)
            y_list=[]
            for i in range(len(Smoothed)):
                y=(i-0)*k+Smoothed[0]
                y_list.append(y)
            ratio=(sum([s-background for s in Smoothed])+0.0000001)/(sum([t-background for t in y_list])+0.0000001)
            return ratio

        def height_ratio(smallONE,bigONE):
            small_original_score_fragment=[]
            for nnp in range(smallONE[2],smallONE[3]+1):
                small_original_score_fragment.append(self.score_list[nnp])
            smallHEIGHT=max(small_original_score_fragment)
            big_original_score_fragment=[]
            for nnp in range(bigONE[2],bigONE[3]+1):
                big_original_score_fragment.append(self.score_list[nnp])
            bigHEIGHT=max(big_original_score_fragment)
            ratio=smallHEIGHT/bigHEIGHT
            return ratio

        def preparation(self,fragment_beginning_index,fragment_ending_index,gap_length,LorR,peak_index,LeftONE,RightONE):
            PCC_score=[]
            CaC_score=[]
            if LorR=='left':
                for midpoint_index in fragment_beginning_index:
                    Original=self.score_list[midpoint_index:fragment_ending_index+1]
                    Smoothed=self.Gaussian_list[midpoint_index:fragment_ending_index+1]
                    pcc_result=Pearson_Correlation_Coefficient(Original,Smoothed)
                    PCC_score.append(pcc_result)
                    cac_result=concavity_and_convexity(Smoothed)
                    CaC_score.append(cac_result)
                    height_ratio_score=height_ratio(LeftONE,RightONE)    ### 小的在前，大的在后
            elif LorR=='right':
                for midpoint_index in fragment_ending_index:
                    Original=self.score_list[fragment_beginning_index:midpoint_index+1]
                    Smoothed=self.Gaussian_list[fragment_beginning_index:midpoint_index+1]
                    pcc_result=Pearson_Correlation_Coefficient(Original,Smoothed)
                    PCC_score.append(pcc_result)
                    cac_result=concavity_and_convexity(Smoothed)
                    CaC_score.append(cac_result)
                    height_ratio_score=height_ratio(RightONE,LeftONE)    ### 小的在前，大的在后
            ### 包含性筛选
            if min(PCC_score)>self.threshold['PCC_independence_1'] and max(CaC_score)>self.threshold['concavity_and_convexity_1'] and gap_length<self.threshold['gap_length_1'] and height_ratio_score>=self.threshold['height_ratio_1']:
                relationship='shifting'
            elif min(PCC_score)>self.threshold['PCC_independence_2'] and max(CaC_score)>self.threshold['concavity_and_convexity_2'] and gap_length<self.threshold['gap_length_2'] and height_ratio_score>=self.threshold['height_ratio_2']:
                relationship='shifting'
            elif min(PCC_score)>self.threshold['PCC_independence_3'] and max(CaC_score)>self.threshold['concavity_and_convexity_3'] and gap_length<self.threshold['gap_length_3'] and height_ratio_score>=self.threshold['height_ratio_3']:
                relationship='shifting'
            elif min(PCC_score)>self.threshold['PCC_independence_4'] and max(CaC_score)>self.threshold['concavity_and_convexity_4'] and gap_length<self.threshold['gap_length_4'] and height_ratio_score>=self.threshold['height_ratio_4']:
                relationship='shifting'
            elif min(PCC_score)>self.threshold['PCC_independence_5'] and max(CaC_score)>self.threshold['concavity_and_convexity_5'] and gap_length<self.threshold['gap_length_5'] and height_ratio_score>=self.threshold['height_ratio_5']:
                relationship='shifting'
            elif min(PCC_score)>self.threshold['PCC_independence_6'] and max(CaC_score)>self.threshold['concavity_and_convexity_6'] and gap_length<self.threshold['gap_length_6'] and height_ratio_score>=self.threshold['height_ratio_6']:
                relationship='shifting'
            else:
                relationship='independent'
            return relationship
            
        def relation_determination(self,peak_index,shoulders_in_a_peak):
            ### shoulders_in_a_peak={-2:inflection_pair_index, -1:inflection_pair_index, 1:inflection_pair_index, 2:inflection_pair_index, ...}
            shoulders_in_a_peak_relation={}
            for position in sorted(shoulders_in_a_peak.keys()):
                inflection_pair_index=shoulders_in_a_peak[position]
                if len(self.inflection_pairs_dict_with_midpoint[inflection_pair_index])>5:
                    midpoints_list_for_pair=self.inflection_pairs_dict_with_midpoint[inflection_pair_index][5:]
                else:
                    midpoints_list_for_pair=[self.inflection_pairs_dict_with_midpoint[inflection_pair_index][:]]
                midpoints_index_list_for_pair=[p[2] for p in midpoints_list_for_pair]+[p[3] for p in midpoints_list_for_pair]
                if position<-1:
                    fragment_beginning_index = midpoints_index_list_for_pair
                    fragment_ending_index = self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position+1]][2]-1
                    gap_length = self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position+1]][2]-self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position]][3]
                    LeftONE = self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position]]
                    RightONE = self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position+1]]
                    independent_or_shifting = preparation(self,fragment_beginning_index,fragment_ending_index,gap_length,'left',peak_index,LeftONE,RightONE)
                elif position==-1:
                    fragment_beginning_index = midpoints_index_list_for_pair
                    fragment_ending_index = self.nucleosome_dict[peak_index][2]-1
                    gap_length = self.nucleosome_dict[peak_index][2]-self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position]][3]
                    LeftONE = self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position]]
                    RightONE = self.nucleosome_dict[peak_index]
                    independent_or_shifting = preparation(self,fragment_beginning_index,fragment_ending_index,gap_length,'left',peak_index,LeftONE,RightONE)
                elif position==1:
                    fragment_beginning_index = self.nucleosome_dict[peak_index][3]+1
                    fragment_ending_index = midpoints_index_list_for_pair
                    gap_length = self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position]][2]-self.nucleosome_dict[peak_index][3]
                    LeftONE = self.nucleosome_dict[peak_index]
                    RightONE = self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position]]
                    independent_or_shifting = preparation(self,fragment_beginning_index,fragment_ending_index,gap_length,'right',peak_index,LeftONE,RightONE)
                elif position>1:
                    fragment_beginning_index = self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position-1]][3]+1
                    fragment_ending_index = midpoints_index_list_for_pair
                    gap_length = self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position]][2]-self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position-1]][3]
                    LeftONE = self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position-1]]
                    RightONE = self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position]]
                    independent_or_shifting = preparation(self,fragment_beginning_index,fragment_ending_index,gap_length,'right',peak_index,LeftONE,RightONE)
                shoulders_in_a_peak_relation[position]=independent_or_shifting
            return shoulders_in_a_peak_relation    ### 输出的结果为一个字典{-2:independent/shifting, -1:independent/shifting, 1:independent/shifting, 2:independent/shifting, ...}
                
        ### Now, check the relationship between two adjecent shoulders OR between a shoulder and its main nucleosome
        ### self.shoulders_sorted_with_main_nucleosomes={peak_index:{-2:inflection_pair_index, -1:inflection_pair_index, 1:inflection_pair_index, 2:inflection_pair_index, ...}, ...}
        ######## 重要记录器 ###################################
        self.more_independent_shoulders=[]
        ######## 重要记录器 ###################################
        shifting_counting=0
        independent_counting=0
        shoulder_candidates_counting=0
        main_nucleosome_integrated_with_shoulder_counting=0
        for peak_index in sorted(self.shoulders_sorted_with_main_nucleosomes.keys()):
            shoulders_in_a_peak=self.shoulders_sorted_with_main_nucleosomes[peak_index]
            ### 调用函数进行独立性计算
            shoulders_in_a_peak_relation=relation_determination(self,peak_index,shoulders_in_a_peak)
            ### 输出的结果为一个字典{-2:independent/shifting, -1:independent/shifting, 1:independent/shifting, 2:independent/shifting, ...}
            one_independent_shoulder={'peak_index':peak_index, 'left_index':'none', 'right_index':'none'}    ### 准备临时记录器
            main_nucleosome_has_been_treated='no'    ### 主峰是否已经过修改的指示器
            for position in sorted(shoulders_in_a_peak_relation.keys()):
                shoulder_candidates_counting=shoulder_candidates_counting+1
                if position<-1:
                    if shoulders_in_a_peak_relation[position]=='independent':
                        independent_counting=independent_counting+1    ### 记录independent
                        if one_independent_shoulder['left_index']=='none':
                            one_independent_shoulder['left_index']=self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position]][2]    ### 找到对应的inflection_pair_index，然后找到左拐点。
                        elif one_independent_shoulder['left_index']!='none':
                            pass
                        one_independent_shoulder['right_index']=self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position]][3]    ### 确定右拐点 ### 完成一个鉴定
                        ### 记录shoulder  [shoulderBC, shoulderEC, shoulderBI, shoulderEI, peak_index]
                        self.more_independent_shoulders.append([one_independent_shoulder['left_index']*10+1,one_independent_shoulder['right_index']*10+1,one_independent_shoulder['left_index'],one_independent_shoulder['right_index'],one_independent_shoulder['peak_index']])
                        one_independent_shoulder={'peak_index':peak_index, 'left_index':'none', 'right_index':'none'}    ###归零
                    elif shoulders_in_a_peak_relation[position]=='shifting':
                        shifting_counting=shifting_counting+1    ### 记录shifting
                        if one_independent_shoulder['left_index']=='none':
                            one_independent_shoulder['left_index']=self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position]][2]
                        elif one_independent_shoulder['left_index']!='none':
                            pass
                elif position==-1:
                    if shoulders_in_a_peak_relation[position]=='independent':
                        independent_counting=independent_counting+1    ### 记录independent
                        if one_independent_shoulder['left_index']=='none':
                            one_independent_shoulder['left_index']=self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position]][2]    ### 找到对应的inflection_pair_index，然后找到左拐点。
                        elif one_independent_shoulder['left_index']!='none':
                            pass
                        one_independent_shoulder['right_index']=self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position]][3]    ### 确定右拐点 ### 完成一个鉴定
                        ### 记录shoulder  [shoulderBC, shoulderEC, shoulderBI, shoulderEI, peak_index]
                        self.more_independent_shoulders.append([one_independent_shoulder['left_index']*10+1,one_independent_shoulder['right_index']*10+1,one_independent_shoulder['left_index'],one_independent_shoulder['right_index'],one_independent_shoulder['peak_index']])
                        one_independent_shoulder={'peak_index':peak_index, 'left_index':'none', 'right_index':'none'}    ###归零
                    elif shoulders_in_a_peak_relation[position]=='shifting':
                        shifting_counting=shifting_counting+1    ### 记录shifting
                        if one_independent_shoulder['left_index']=='none':
                            one_independent_shoulder['left_index']=self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position]][2]
                        elif one_independent_shoulder['left_index']!='none':
                            pass
                        if max(shoulders_in_a_peak_relation.keys())==-1:    ### 只有左边的shoulder，右边没了。
                            one_independent_shoulder['right_index']=self.nucleosome_dict[peak_index][3]    ### 确定右拐点 ### 完成一个鉴定
                            ### 修改记录self.nucleosome_dict={peak_index:[pairBC, pairEC, pairBI, pairEI, pair_index, 'height', 'area', [peakBC, maxBC, maxEC, peakEC, peakBI, maxBI, maxEI, peakEI, peak_index]], ...}
                            self.nucleosome_dict[peak_index][2]=one_independent_shoulder['left_index']
                            self.nucleosome_dict[peak_index][3]=one_independent_shoulder['right_index']
                            self.nucleosome_dict[peak_index][0]=one_independent_shoulder['left_index']*10+1
                            self.nucleosome_dict[peak_index][1]=one_independent_shoulder['right_index']*10+1
                            main_nucleosome_integrated_with_shoulder_counting=main_nucleosome_integrated_with_shoulder_counting+1
                            self.nucleosome_dict[peak_index].append('Main_nucleosome:integrated')
                            main_nucleosome_has_been_treated='yes'    ### 注明修改主峰。
                            one_independent_shoulders={'peak_index':peak_index, 'left_index':'none', 'right_index':'none'}    ###归零
                        elif max(shoulders_in_a_peak_relation.keys())>=1:
                            pass
                elif position==1:
                    if shoulders_in_a_peak_relation[position]=='independent':
                        independent_counting=independent_counting+1    ### 记录independent
                        if one_independent_shoulder['left_index']!='none':    ### 存在（未完成的，正在进行中的）跨主峰合并
                            one_independent_shoulder['right_index']=self.nucleosome_dict[peak_index][3]    ### 确定右拐点 ### 完成一个鉴定
                            ### 修改记录self.nucleosome_dict={peak_index:[pairBC, pairEC, pairBI, pairEI, pair_index, 'height', 'area', [peakBC, maxBC, maxEC, peakEC, peakBI, maxBI, maxEI, peakEI, peak_index]], ...}
                            self.nucleosome_dict[peak_index][2]=one_independent_shoulder['left_index']
                            self.nucleosome_dict[peak_index][3]=one_independent_shoulder['right_index']
                            self.nucleosome_dict[peak_index][0]=one_independent_shoulder['left_index']*10+1
                            self.nucleosome_dict[peak_index][1]=one_independent_shoulder['right_index']*10+1
                            main_nucleosome_integrated_with_shoulder_counting=main_nucleosome_integrated_with_shoulder_counting+1
                            self.nucleosome_dict[peak_index].append('Main_nucleosome:integrated')
                            main_nucleosome_has_been_treated='yes'    ### 注明修改主峰。
                            one_independent_shoulder={'peak_index':peak_index, 'left_index':'none', 'right_index':'none'}    ###归零
                        elif one_independent_shoulder['left_index']=='none':    ### 说明主峰的左右两侧均无牵连，主峰保持不变。
                            ### 此处主峰无需修改。
                            main_nucleosome_has_been_treated='yes'
                            ### 此处主峰无需修改。
                        ### 启动新的shoulder判定。
                        one_independent_shoulder['left_index']=self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position]][2]
                        if max(shoulders_in_a_peak_relation.keys())==position:    ### 如果当前shoulder是这个peak下的最后一个shoulder了
                            one_independent_shoulder['right_index']=self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position]][3]    ### 确定右拐点 ### 完成一个鉴定
                            ### 记录shoulder  [shoulderBC, shoulderEC, shoulderBI, shoulderEI, peak_index]
                            self.more_independent_shoulders.append([one_independent_shoulder['left_index']*10+1,one_independent_shoulder['right_index']*10+1,one_independent_shoulder['left_index'],one_independent_shoulder['right_index'],one_independent_shoulder['peak_index']])
                            one_independent_shoulder={'peak_index':peak_index, 'left_index':'none', 'right_index':'none'}    ###归零
                        elif max(shoulders_in_a_peak_relation.keys())>position:
                            pass
                    elif shoulders_in_a_peak_relation[position]=='shifting':
                        shifting_counting=shifting_counting+1    ### 记录shifting
                        if one_independent_shoulder['left_index']!='none':    ### 存在（未完成的，正在进行中的）跨主峰合并
                            pass
                        elif one_independent_shoulder['left_index']=='none':    ### 说明主峰的左拐点为跨主峰合并的起点
                            one_independent_shoulder['left_index']=self.nucleosome_dict[peak_index][2]    ### 确定左拐点
                        if max(shoulders_in_a_peak_relation.keys())==position:    ### 如果当前shoulder是这个peak下的最后一个shoulder了
                            one_independent_shoulder['right_index']=self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position]][3]    ### 确定右拐点 ### 完成一个鉴定
                            ### 修改记录self.nucleosome_dict={peak_index:[pairBC, pairEC, pairBI, pairEI, pair_index, 'height', 'area', [peakBC, maxBC, maxEC, peakEC, peakBI, maxBI, maxEI, peakEI, peak_index]], ...}
                            self.nucleosome_dict[peak_index][2]=one_independent_shoulder['left_index']
                            self.nucleosome_dict[peak_index][3]=one_independent_shoulder['right_index']
                            self.nucleosome_dict[peak_index][0]=one_independent_shoulder['left_index']*10+1
                            self.nucleosome_dict[peak_index][1]=one_independent_shoulder['right_index']*10+1
                            main_nucleosome_integrated_with_shoulder_counting=main_nucleosome_integrated_with_shoulder_counting+1
                            self.nucleosome_dict[peak_index].append('Main_nucleosome:integrated')
                            main_nucleosome_has_been_treated='yes'    ### 注明修改主峰。
                            one_independent_shoulder={'peak_index':peak_index, 'left_index':'none', 'right_index':'none'}    ###归零
                        elif max(shoulders_in_a_peak_relation.keys())>position:
                            pass
                elif position>1:
                    if shoulders_in_a_peak_relation[position]=='independent':
                        independent_counting=independent_counting+1    ### 记录independent
                        if one_independent_shoulder['left_index']!='none':    ### 存在（未完成的，正在进行中的）合并
                            one_independent_shoulder['right_index']=self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position-1]][3]    ### 确定右拐点 ### 完成一个鉴定
                            if main_nucleosome_has_been_treated=='yes':    ### 如果主峰已经过修改。
                                ### 记录shoulder  [shoulderBC, shoulderEC, shoulderBI, shoulderEI, peak_index]
                                self.more_independent_shoulders.append([one_independent_shoulder['left_index']*10+1,one_independent_shoulder['right_index']*10+1,one_independent_shoulder['left_index'],one_independent_shoulder['right_index'],one_independent_shoulder['peak_index']])
                            elif main_nucleosome_has_been_treated=='no':    ### 如果主峰未经过修改。
                                ### 修改记录self.nucleosome_dict={peak_index:[pairBC, pairEC, pairBI, pairEI, pair_index, 'height', 'area', [peakBC, maxBC, maxEC, peakEC, peakBI, maxBI, maxEI, peakEI, peak_index]], ...}
                                self.nucleosome_dict[peak_index][2]=one_independent_shoulder['left_index']
                                self.nucleosome_dict[peak_index][3]=one_independent_shoulder['right_index']
                                self.nucleosome_dict[peak_index][0]=one_independent_shoulder['left_index']*10+1
                                self.nucleosome_dict[peak_index][1]=one_independent_shoulder['right_index']*10+1
                                main_nucleosome_integrated_with_shoulder_counting=main_nucleosome_integrated_with_shoulder_counting+1
                                self.nucleosome_dict[peak_index].append('Main_nucleosome:integrated')
                                main_nucleosome_has_been_treated='yes'    ### 注明修改主峰。
                            one_independent_shoulder={'peak_index':peak_index, 'left_index':'none', 'right_index':'none'}    ###归零
                        elif one_independent_shoulder['left_index']=='none':
                            pass
                        ### 启动新的shoulder判定。
                        one_independent_shoulder['left_index']=self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position]][2]
                        if max(shoulders_in_a_peak_relation.keys())==position:    ### 如果当前shoulder是这个peak下的最后一个shoulder了
                            one_independent_shoulder['right_index']=self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position]][3]    ### 确定右拐点 ### 完成一个鉴定
                            ### 记录shoulder  [shoulderBC, shoulderEC, shoulderBI, shoulderEI, peak_index]
                            self.more_independent_shoulders.append([one_independent_shoulder['left_index']*10+1,one_independent_shoulder['right_index']*10+1,one_independent_shoulder['left_index'],one_independent_shoulder['right_index'],one_independent_shoulder['peak_index']])
                            one_independent_shoulder={'peak_index':peak_index, 'left_index':'none', 'right_index':'none'}    ###归零
                        elif max(shoulders_in_a_peak_relation.keys())>position:
                            pass
                    elif shoulders_in_a_peak_relation[position]=='shifting':
                        shifting_counting=shifting_counting+1    ### 记录shifting
                        if one_independent_shoulder['left_index']!='none':
                            pass
                        elif one_independent_shoulder['left_index']=='none':
                            pass
                        if max(shoulders_in_a_peak_relation.keys())==position:    ### 如果当前shoulder是这个peak下的最后一个shoulder了
                            one_independent_shoulder['right_index']=self.inflection_pairs_dict_with_midpoint[shoulders_in_a_peak[position]][3]    ### 确定右拐点 ### 完成一个鉴定
                            if main_nucleosome_has_been_treated=='yes':    ### 如果主峰已经过修改。
                                ### 记录shoulder  [shoulderBC, shoulderEC, shoulderBI, shoulderEI, peak_index]
                                self.more_independent_shoulders.append([one_independent_shoulder['left_index']*10+1,one_independent_shoulder['right_index']*10+1,one_independent_shoulder['left_index'],one_independent_shoulder['right_index'],one_independent_shoulder['peak_index']])
                            elif main_nucleosome_has_been_treated=='no':    ### 如果主峰未经过修改。
                                ### 修改记录nucleosome_dict={peak_index:[pairBC, pairEC, pairBI, pairEI, pair_index, 'height', 'area', [peakBC, maxBC, maxEC, peakEC, peakBI, maxBI, maxEI, peakEI, peak_index]], ...}
                                self.nucleosome_dict[peak_index][2]=one_independent_shoulder['left_index']
                                self.nucleosome_dict[peak_index][3]=one_independent_shoulder['right_index']
                                self.nucleosome_dict[peak_index][0]=one_independent_shoulder['left_index']*10+1
                                self.nucleosome_dict[peak_index][1]=one_independent_shoulder['right_index']*10+1
                                main_nucleosome_integrated_with_shoulder_counting=main_nucleosome_integrated_with_shoulder_counting+1
                                self.nucleosome_dict[peak_index].append('Main_nucleosome:integrated')
                                main_nucleosome_has_been_treated='yes'    ### 注明修改主峰。
                            one_independent_shoulder={'peak_index':peak_index, 'left_index':'none', 'right_index':'none'}    ###归零
                        elif max(shoulders_in_a_peak_relation.keys())>position:
                            pass

        ### Now, intergrating main nucleosome with the independent shoulders.
        ### self.more_independent_shoulders=[[shoulderBC, shoulderEC, shoulderBI, shoulderEI, peak_index], ... ] 已完备
        ### self.nucleosome_dict={peak_index:[pairBC, pairEC, pairBI, pairEI, pair_index, 'height', 'area', [peakBC, maxBC, maxEC, peakEC, peakBI, maxBI, maxEI, peakEI, peak_index]], ...} 已完备
        ### 开始整合
        ######## 重要记录器 ###################################
        self.nucleosome_integrated=[]
        ######## 重要记录器 ###################################
        shoulder_counting=0
        shoulder_counting_suplimit=len(self.more_independent_shoulders)-1
        peak_index=1
        if len( self.nucleosome_dict.keys() ) > 0:
            peak_index_suplimit = sorted( list( self.nucleosome_dict.keys() ) )[-1]  ##peak_index_suplimit=list(self.nucleosome_dict.keys())[-1]
        else:
            peak_index_suplimit = 0
        total_shoulders=0
        total_main_nucleosomes=0
        isolated_main_nucleosome_counting=0
        if len( self.more_independent_shoulders ) > 0:
            for bp in range(len(self.score_table)):
                if self.more_independent_shoulders[shoulder_counting][0]==self.score_table[bp][0]:
                    ### Record the shoulder.
                    self.nucleosome_integrated.append([self.more_independent_shoulders[shoulder_counting][0],self.more_independent_shoulders[shoulder_counting][1],self.more_independent_shoulders[shoulder_counting][2],self.more_independent_shoulders[shoulder_counting][3],'height','area',[self.peak_dict[peak_index][0],self.peak_dict[peak_index][1],self.peak_dict[peak_index][2],self.peak_dict[peak_index][3],self.peak_dict[peak_index][4],self.peak_dict[peak_index][5],self.peak_dict[peak_index][6],self.peak_dict[peak_index][7],peak_index]])
                    self.nucleosome_integrated[-1].append('Independent_shoulder')
                    total_shoulders=total_shoulders+1    ### Counting
                    if shoulder_counting<shoulder_counting_suplimit:
                        shoulder_counting=shoulder_counting+1
                    else:
                        pass
                else:
                    while peak_index not in self.nucleosome_dict.keys():
                        peak_index=peak_index+1
                    if self.nucleosome_dict[peak_index][0]==self.score_table[bp][0]:
                        ### Record the main nucleosome.
                        self.nucleosome_integrated.append(self.nucleosome_dict[peak_index])
                        self.nucleosome_integrated[-1].pop(4)
                        if self.nucleosome_integrated[-1][-1]=='Main_nucleosome:integrated':
                            pass
                        elif self.nucleosome_integrated[-1][-1]!='Main_nucleosome:integrated':
                            self.nucleosome_integrated[-1].append('Main_nucleosome:isolated')
                            isolated_main_nucleosome_counting=isolated_main_nucleosome_counting+1
                        total_main_nucleosomes=total_main_nucleosomes+1
                        if peak_index<peak_index_suplimit:
                            peak_index=peak_index+1
                        else:
                            pass
        else:
            for bp in range(len(self.score_table)):
                #if self.more_independent_shoulders[shoulder_counting][0]==self.score_table[bp][0]:
                #    ### Record the shoulder.
                #    self.nucleosome_integrated.append([self.more_independent_shoulders[shoulder_counting][0],self.more_independent_shoulders[shoulder_counting][1],self.more_independent_shoulders[shoulder_counting][2],self.more_independent_shoulders[shoulder_counting][3],'height','area',[self.peak_dict[peak_index][0],self.peak_dict[peak_index][1],self.peak_dict[peak_index][2],self.peak_dict[peak_index][3],self.peak_dict[peak_index][4],self.peak_dict[peak_index][5],self.peak_dict[peak_index][6],self.peak_dict[peak_index][7],peak_index]])
                #    self.nucleosome_integrated[-1].append('Independent_shoulder')
                #    total_shoulders=total_shoulders+1    ### Counting
                #    if shoulder_counting<shoulder_counting_suplimit:
                #        shoulder_counting=shoulder_counting+1
                #    else:
                #        pass
                #else:
                if len(self.nucleosome_dict)==0:
                    continue
                else:
                    while peak_index not in self.nucleosome_dict.keys():
                        peak_index=peak_index+1
                if self.nucleosome_dict[peak_index][0]==self.score_table[bp][0]:
                    ### Record the main nucleosome.
                    self.nucleosome_integrated.append(self.nucleosome_dict[peak_index])
                    self.nucleosome_integrated[-1].pop(4)
                    if self.nucleosome_integrated[-1][-1]=='Main_nucleosome:integrated':
                        pass
                    elif self.nucleosome_integrated[-1][-1]!='Main_nucleosome:integrated':
                        self.nucleosome_integrated[-1].append('Main_nucleosome:isolated')
                        isolated_main_nucleosome_counting=isolated_main_nucleosome_counting+1
                    total_main_nucleosomes=total_main_nucleosomes+1
                    if peak_index<peak_index_suplimit:
                        peak_index=peak_index+1
                    else:
                        pass
        print('\tChecking ==> Total shoulder candidates counting:  ',shoulder_candidates_counting)
        print('\tChecking ==> Dynamic shifting shoulder candidates:  ',shifting_counting)
        print('\tChecking ==> Independent shoulder candidates:  ',independent_counting)
        print('\tChecking ==> ',shifting_counting,'+',independent_counting,'=',shifting_counting+independent_counting)
        print('\tChecking ==> Total independent shoulders:  ',total_shoulders)
        print('\tChecking ==> Total main nucleosomes:  ',total_main_nucleosomes)
        print('\tChecking ==> Integrated main nucleosomes:  ',main_nucleosome_integrated_with_shoulder_counting)
        print('\tChecking ==> Isolated main nucleosomes:  ',isolated_main_nucleosome_counting)
        print('\tChecking ==> ',isolated_main_nucleosome_counting,'+',main_nucleosome_integrated_with_shoulder_counting,'=',isolated_main_nucleosome_counting+main_nucleosome_integrated_with_shoulder_counting)
        print('\tChecking ==> Total main nucleosomes and independent shoulders:  ',len(self.nucleosome_integrated))
        print('\tChecking ==> ',total_main_nucleosomes,'+',total_shoulders,'=',total_main_nucleosomes+total_shoulders)
        ######## 重要记录器 ###################################
        self.final_nucleosome=[]
        ######## 重要记录器 ###################################
        for i in range(len(self.nucleosome_integrated)):    ### 新的记录
            self.final_nucleosome.append(self.nucleosome_integrated[i][0:6]+[self.nucleosome_integrated[i][6]]+[self.nucleosome_integrated[i][6]]+[self.nucleosome_integrated[i][7]])
        print('...... Precise nucleosome positioning is finished.','\t',time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))


            
    def adjust_border(self):
        print('Adjust the border of every inflection pair with secondary inflection detection ......')
        ### Adjust the border of every inflection pair with secondary inflection detection.
        ### self.nucleosome_dict={peak_index:[pairBC, pairEC, pairBI, pairEI, 'height', 'area', [peakBC, maxBC, maxEC, peakEC, peakBI, maxBI, maxEI, peakEI, peak_index], main/shoulder], ...} 已完备
        def influence_coefficient(self,ii,influence_from_left_or_right):
            ### Calculate distance between border and distance between center
            if influence_from_left_or_right=='left':
                influence_index=ii-1
                distance=self.nucleosome_integrated[ii][2]-self.nucleosome_integrated[influence_index][3]    ### distance
                if self.nucleosome_integrated[ii][-1]=='Main_nucleosome:integrated' or self.nucleosome_integrated[ii][-1]=='Main_nucleosome:isolated':
                    center_1=self.nucleosome_integrated[ii][6][5]
                elif self.nucleosome_integrated[ii][-1]=='Independent_shoulder':
                    center_1=(self.nucleosome_integrated[ii][2]+self.nucleosome_integrated[ii][3])/2
                if self.nucleosome_integrated[influence_index][-1]=='Main_nucleosome:integrated' or self.nucleosome_integrated[influence_index][-1]=='Main_nucleosome:isolated':
                    center_2=self.nucleosome_integrated[influence_index][6][6]
                elif self.nucleosome_integrated[influence_index][-1]=='Independent_shoulder':
                    center_2=(self.nucleosome_integrated[influence_index][2]+self.nucleosome_integrated[influence_index][3])/2
                distance_center=center_1-center_2    ### distance from center
            elif influence_from_left_or_right=='right':
                influence_index=ii+1
                distance=self.nucleosome_integrated[influence_index][2]-self.nucleosome_integrated[ii][3]    ### distance
                if self.nucleosome_integrated[influence_index][-1]=='Main_nucleosome:integrated' or self.nucleosome_integrated[influence_index][-1]=='Main_nucleosome:isolated':
                    center_1=self.nucleosome_integrated[influence_index][6][5]
                elif self.nucleosome_integrated[influence_index][-1]=='Independent_shoulder':
                    center_1=(self.nucleosome_integrated[influence_index][2]+self.nucleosome_integrated[influence_index][3])/2
                if self.nucleosome_integrated[ii][-1]=='Main_nucleosome:integrated' or self.nucleosome_integrated[ii][-1]=='Main_nucleosome:isolated':
                    center_2=self.nucleosome_integrated[ii][6][6]
                elif self.nucleosome_integrated[ii][-1]=='Independent_shoulder':
                    center_2=(self.nucleosome_integrated[ii][2]+self.nucleosome_integrated[ii][3])/2
                distance_center=center_1-center_2    ### distance from center
            ### height, AUC, smoothedheight ### self
            temp_height_self=[]
            temp_smoothedheight_self=[]
            for nnp in range(self.nucleosome_integrated[ii][2],self.nucleosome_integrated[ii][3]+1):
                temp_height_self.append(self.score_table[nnp][1])
                temp_smoothedheight_self.append(self.score_table[nnp][2])
            height_self=max(temp_height_self)
            area_self=sum(temp_height_self)
            smoothedheight_self=max(temp_smoothedheight_self)
            smoothedarea_self=sum(temp_smoothedheight_self)
            ### height, AUC, smoothedheight ### influence
            temp_height_influence=[]
            temp_smoothedheight_influence=[]
            for nnp in range(self.nucleosome_integrated[influence_index][2],self.nucleosome_integrated[influence_index][3]+1):
                temp_height_influence.append(self.score_table[nnp][1])
                temp_smoothedheight_influence.append(self.score_table[nnp][2])
            height_influence=max(temp_height_influence)
            area_influence=sum(temp_height_influence)
            smoothedheight_influence=max(temp_smoothedheight_influence)
            smoothedarea_influence=sum(temp_smoothedheight_influence)
            ### Comparing
            if height_self > 0:                #### new code #### 20150821
                coefficient=(height_influence/height_self)*10/distance
            elif height_self <= 0:
                coefficient=(height_influence/1)*10/distance
            coefficient_absolute=height_influence*10/distance
            if height_influence<height_self or smoothedheight_influence<smoothedheight_self:
                effect='no'
            elif distance>self.threshold['influence_coefficient_distance'] or distance_center>self.threshold['influence_coefficient_distance_center']:
                effect='no'
            else:
                if coefficient<self.threshold['influence_coefficient_cutoff'] or coefficient_absolute<self.threshold['influence_coefficient_absolute_cutoff']:
                    effect='no'
                else:
                    effect='yes'
            return [effect,coefficient]

        influence_list=[]
        for i in range(len(self.nucleosome_integrated)):
            influence_list.append([])
            peak_index=self.nucleosome_integrated[i][-2]
            if i==0:
                right_effect=influence_coefficient(self,i,'right')
                if right_effect[0]=='yes':
                    influence_list[-1].append('right')
                else:
                    pass
            elif i>0 and i<len(self.nucleosome_integrated)-1:
                right_effect=influence_coefficient(self,i,'right')
                left_effect=influence_coefficient(self,i,'left')
                if right_effect[0]=='yes':
                    influence_list[-1].append('right')
                else:
                    pass
                if left_effect[0]=='yes':
                    influence_list[-1].append('left')
                else:
                    pass
            elif i==len(self.nucleosome_integrated)-1:
                left_effect=influence_coefficient(self,i,'left')
                if left_effect[0]=='yes':
                    influence_list[-1].append('left')
                else:
                    pass

        if len( self.nucleosome_dict.keys() ) > 0:
            peak_index_suplimit = sorted( list(self.nucleosome_dict.keys()) )[-1] ###peak_index_suplimit=list(self.nucleosome_dict.keys())[-1]
        else:
            peak_index_suplimit = 0
        for j in range(len(self.nucleosome_integrated)):
            peak_index=self.nucleosome_integrated[j][-2][-1]
            adjacent_influence=influence_list[j]
            ### 收集周围的小拐点
            secondary_canditates=[]
            if peak_index==1:
                peak_index_list=[peak_index,peak_index+1]
            elif peak_index>1 and peak_index<peak_index_suplimit:
                peak_index_list=[peak_index-1,peak_index,peak_index+1]
            elif peak_index==peak_index_suplimit:
                peak_index_list=[peak_index-1,peak_index]
            for peak_index_number in peak_index_list:
                if peak_index_number in self.secondary_inflection_pairs_dict_with_peak.keys():
                    for candidate in self.secondary_inflection_pairs_dict_with_peak[peak_index_number]:
                        if candidate not in secondary_canditates:
                            secondary_canditates.append(candidate)
                        else:
                            pass
                else:
                    pass
            ### 选择最适合的小拐点
            temp_left_index=0
            temp_left=secondary_canditates[0][2]-self.nucleosome_integrated[j][2]
            temp_right_index=len(secondary_canditates)-1
            temp_right=secondary_canditates[len(secondary_canditates)-1][3]-self.nucleosome_integrated[j][3]
            for k in range(len(secondary_canditates)):
                if abs(secondary_canditates[k][2]-self.nucleosome_integrated[j][2])<abs(temp_left):
                    temp_left_index=k
                    temp_left=secondary_canditates[k][2]-self.nucleosome_integrated[j][2]
                    if 'left' in adjacent_influence:
                        if secondary_canditates[k][2]<self.nucleosome_integrated[j][2] and secondary_canditates[k][2]>(self.nucleosome_integrated[j-1][3]+self.nucleosome_integrated[j][2])/2:
                            self.nucleosome_integrated[j][0]=secondary_canditates[temp_left_index][0]
                            self.nucleosome_integrated[j][2]=secondary_canditates[temp_left_index][2]
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
                if abs(secondary_canditates[len(secondary_canditates)-1-k][3]-self.nucleosome_integrated[j][3])<abs(temp_right):
                    temp_right_index=len(secondary_canditates)-1-k
                    temp_right=secondary_canditates[len(secondary_canditates)-1-k][3]-self.nucleosome_integrated[j][3]
                    if 'right' in adjacent_influence:
                        if secondary_canditates[len(secondary_canditates)-1-k][3]>self.nucleosome_integrated[j][3] and secondary_canditates[len(secondary_canditates)-1-k][3]<(self.nucleosome_integrated[j][3]+self.nucleosome_integrated[j+1][2])/2:
                            self.nucleosome_integrated[j][1]=secondary_canditates[temp_right_index][1]
                            self.nucleosome_integrated[j][3]=secondary_canditates[temp_right_index][3]
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
        ######## 重要记录器 ###################################
        self.final_nucleosome=[]
        ######## 重要记录器 ###################################
        for i in range(len(self.nucleosome_integrated)):    ### 新的记录
            self.final_nucleosome.append(self.nucleosome_integrated[i][0:6]+[self.nucleosome_integrated[i][6]]+[self.nucleosome_integrated[i][6]]+[self.nucleosome_integrated[i][7]])
        print('...... Border adjustment is finished.','\t',time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))



    def filter_results(self):            
        ### Filter and record the main nucleosome.
        print('Filter and annotate the nucleosomes ......')
        ### self.nucleosome_integrated=[[pairBC, pairEC, pairBI, pairEI, 'height', 'area', [peakBC, maxBC, maxEC, peakEC, peakBI, maxBI, maxEI, peakEI, peak_index], main/shoulder], ...]
        interval_property_list=[]
        for n in range(len(self.nucleosome_integrated)-1):
            one_interval={'Distance_between_center':'none',  'Interval_length':'none',  'Original_Interval_Depth':'none',  'Smoothed_Interval_Depth':'none',
                          'Left_OriginalHeight':'none',   'Left_OriginalAUC':'none',   'Left_SmoothedHeight':'none',   'Left_SmoothedAUC':'none',
                          'Right_OriginalHeight':'none',  'Right_OriginalAUC':'none',  'Right_SmoothedHeight':'none',  'Right_SmoothedAUC':'none',
                          'Smoothed_Interval_Depth_percentage':'none',
                          'HeightRatio':'none',
                          'Merge':'no',
                          'Noise_on_left_or_right':'none'}
            one_interval['Distance_between_center']=(self.nucleosome_integrated[n+1][0]+self.nucleosome_integrated[n+1][1])/2-(self.nucleosome_integrated[n][0]+self.nucleosome_integrated[n][1])/2
            one_interval['Interval_length']=self.nucleosome_integrated[n+1][0]-self.nucleosome_integrated[n][1]    ### 关键
            Interval_original_score_fragment=[]
            Interval_smoothed_score_fragment=[]
            for nnq in range(self.nucleosome_integrated[n][3]+1,self.nucleosome_integrated[n+1][2]):
                Interval_original_score_fragment.append(self.score_list[nnq])
                Interval_smoothed_score_fragment.append(self.Gaussian_list[nnq])
            one_interval['Original_Interval_Depth']=min(Interval_original_score_fragment)
            one_interval['Smoothed_Interval_Depth']=min(Interval_smoothed_score_fragment)
            ####
            Left_original_score_fragment=[]
            Left_smoothed_score_fragment=[]
            for nnp in range(self.nucleosome_integrated[n][2],self.nucleosome_integrated[n][3]+1):
                Left_original_score_fragment.append(self.score_list[nnp])
                Left_smoothed_score_fragment.append(self.Gaussian_list[nnp])
            one_interval['Left_OriginalHeight']=max(Left_original_score_fragment)
            one_interval['Left_OriginalAUC']=sum(Left_original_score_fragment)
            one_interval['Left_SmoothedHeight']=max(Left_smoothed_score_fragment)
            one_interval['Left_SmoothedAUC']=sum(Left_smoothed_score_fragment)
            ####
            Right_original_score_fragment=[]
            Right_smoothed_score_fragment=[]
            for nnp in range(self.nucleosome_integrated[n+1][2],self.nucleosome_integrated[n+1][3]+1):
                Right_original_score_fragment.append(self.score_list[nnp])
                Right_smoothed_score_fragment.append(self.Gaussian_list[nnp])
            one_interval['Right_OriginalHeight']=max(Right_original_score_fragment)
            one_interval['Right_OriginalAUC']=sum(Right_original_score_fragment)
            one_interval['Right_SmoothedHeight']=max(Right_smoothed_score_fragment)
            one_interval['Right_SmoothedAUC']=sum(Right_smoothed_score_fragment)
            ####
            one_interval['Smoothed_Interval_Depth_percentage']=2*one_interval['Smoothed_Interval_Depth']/(one_interval['Left_SmoothedHeight']+one_interval['Right_SmoothedHeight'])    ### 关键
            one_interval['HeightRatio']=min(one_interval['Left_OriginalHeight'],one_interval['Right_OriginalHeight'])/max(one_interval['Left_OriginalHeight'],one_interval['Right_OriginalHeight'])    ### 关键
            ########           
            if one_interval['Distance_between_center']<self.threshold['merging_center_distance']:
                if max(one_interval['Left_OriginalHeight'],one_interval['Right_OriginalHeight'])<self.threshold['merging_height_watershed']:
                    if one_interval['HeightRatio']>self.threshold['merging_height_ratio_1'] and one_interval['Interval_length']<=self.threshold['merging_gap_1'] and one_interval['Smoothed_Interval_Depth_percentage']>self.threshold['merging_percentage_1']:
                        one_interval['Merge']='yes'
                    elif one_interval['HeightRatio']>self.threshold['merging_height_ratio_3'] and one_interval['Interval_length']<=self.threshold['merging_gap_3'] and one_interval['Smoothed_Interval_Depth_percentage']>self.threshold['merging_percentage_3']:
                        one_interval['Merge']='yes'
                    elif one_interval['HeightRatio']>self.threshold['merging_height_ratio_4'] and one_interval['Interval_length']<=self.threshold['merging_gap_4'] and one_interval['Smoothed_Interval_Depth_percentage']>self.threshold['merging_percentage_4']:
                        one_interval['Merge']='yes'
                    else:
                        one_interval['Merge']='no'
                elif max(one_interval['Left_OriginalHeight'],one_interval['Right_OriginalHeight'])>=self.threshold['merging_height_watershed']:
                    if one_interval['HeightRatio']>self.threshold['merging_height_ratio_2'] and one_interval['Interval_length']<=self.threshold['merging_gap_2'] and one_interval['Smoothed_Interval_Depth_percentage']>self.threshold['merging_percentage_2']:
                        one_interval['Merge']='yes'
                    elif one_interval['HeightRatio']>self.threshold['merging_height_ratio_3'] and one_interval['Interval_length']<=self.threshold['merging_gap_3'] and one_interval['Smoothed_Interval_Depth_percentage']>self.threshold['merging_percentage_3']:
                        one_interval['Merge']='yes'
                    elif one_interval['HeightRatio']>self.threshold['merging_height_ratio_4'] and one_interval['Interval_length']<=self.threshold['merging_gap_4'] and one_interval['Smoothed_Interval_Depth_percentage']>self.threshold['merging_percentage_4']:
                        one_interval['Merge']='yes'
                    else:
                        one_interval['Merge']='no'
            else:
                one_interval['Merge']='no'
            interval_property_list.append(one_interval)
        ### Checking recording.
        print('\tChecking ==> Number of interval:  ',len(interval_property_list))
        for n in range(len(self.nucleosome_integrated)-2):
            if interval_property_list[n]['Right_OriginalHeight']==interval_property_list[n+1]['Left_OriginalHeight'] and interval_property_list[n]['Right_OriginalAUC']==interval_property_list[n+1]['Left_OriginalAUC'] and interval_property_list[n]['Right_SmoothedHeight']==interval_property_list[n+1]['Left_SmoothedHeight'] and interval_property_list[n]['Right_SmoothedAUC']==interval_property_list[n+1]['Left_SmoothedAUC']:
                pass
            else:
                print('ERROR in interval mapping:  ',n,' and ',n+1)

        ### 进行新的一轮合并
        def merge_and_annotate(self,interval_property_list):
            merging_choice_list=['none']*len(self.nucleosome_integrated)
            for i in range(len(self.nucleosome_integrated)):
                if i==0:
                    if interval_property_list[i]['Merge']=='yes':
                        merging_choice_list[i]='right'
                    else:
                        pass
                elif i>0 and i<len(self.nucleosome_integrated)-1:
                    if interval_property_list[i]['Merge']=='yes' and interval_property_list[i-1]['Merge']=='no':
                        merging_choice_list[i]='right'
                    elif interval_property_list[i]['Merge']=='no' and interval_property_list[i-1]['Merge']=='yes':
                        merging_choice_list[i]='left'
                    elif interval_property_list[i]['Merge']=='yes' and interval_property_list[i-1]['Merge']=='yes':
                        if interval_property_list[i]['Smoothed_Interval_Depth']>interval_property_list[i-1]['Smoothed_Interval_Depth']:
                            merging_choice_list[i]='right'
                        elif interval_property_list[i]['Smoothed_Interval_Depth']<interval_property_list[i-1]['Smoothed_Interval_Depth']:
                            merging_choice_list[i]='left'
                        elif interval_property_list[i]['Smoothed_Interval_Depth']==interval_property_list[i-1]['Smoothed_Interval_Depth']:
                            if interval_property_list[i]['Interval_length']<interval_property_list[i-1]['Interval_length']:
                                merging_choice_list[i]='right'
                            elif interval_property_list[i]['Interval_length']>interval_property_list[i-1]['Interval_length']:
                                merging_choice_list[i]='left'
                            elif interval_property_list[i]['Interval_length']==interval_property_list[i-1]['Interval_length']:
                                pass
                    elif interval_property_list[i]['Merge']=='no' and interval_property_list[i-1]['Merge']=='no':
                        pass
                elif i==len(self.nucleosome_integrated)-1:
                    if interval_property_list[i-1]['Merge']=='yes':
                        merging_choice_list[i]='left'
                    else:
                        pass
            #### 调试专用 #############################
            ####tempfile=open(self.outputfile_path+'MergingChoice.txt','w')
            ####for i in range(len(self.nucleosome_integrated)):
            ####    tempfile.write('nucleosome_candidates:'+str(i+1)+'\t'+'peak_index:'+str(self.nucleosome_integrated[i][6][8])+'\t'+merging_choice_list[i]+'\n')
            ####    if i<len(self.nucleosome_integrated)-1:
            ####        tempfile.write('\t'+'interval:'+str(i+1)+'->'+str(i+2)+'\t'+interval_property_list[i]['Merge']+'\n')
            ####    else:
            ####        pass
            ####tempfile.close()
            #### 调试专用 #############################
            merging_counting=0
            unmerging_counting=0
            merging_switch='off'
            final_nucleosome_temp_recorder=[]    ### 记录器
            one_nucleosome_for_final_nucleosome_temp_recorder=['beginning pairBC', 'ending pairEC', 'beginning pairBI', 'ending pairEI', 'height', 'area', '[beginning peak region]', '[ending peak region]', 'main/shoulder']
            ### self.nucleosome_integrated=[[pairBC, pairEC, pairBI, pairEI, 'height', 'area', [peakBC, maxBC, maxEC, peakEC, peakBI, maxBI, maxEI, peakEI, peak_index], main/shoulder], ...]
            for i in range(len(self.nucleosome_integrated)):
                if merging_choice_list[i]=='right':
                    if merging_switch=='off':    ### 前面一个不是right
                        one_nucleosome_for_final_nucleosome_temp_recorder[0]=self.nucleosome_integrated[i][0]
                        one_nucleosome_for_final_nucleosome_temp_recorder[2]=self.nucleosome_integrated[i][2]
                        one_nucleosome_for_final_nucleosome_temp_recorder[6]=self.nucleosome_integrated[i][6]
                        one_nucleosome_for_final_nucleosome_temp_recorder[8]=self.nucleosome_integrated[i][7]
                        merging_switch='on'
                    elif merging_switch=='on':    ### 前面一个也是right
                        final_nucleosome_temp_recorder.append(self.nucleosome_integrated[i-1][0:6]+[self.nucleosome_integrated[i-1][6]]+[self.nucleosome_integrated[i-1][6]]+[self.nucleosome_integrated[i-1][7]])    ### 记录前一个
                        unmerging_counting=unmerging_counting+1
                        one_nucleosome_for_final_nucleosome_temp_recorder[0]=self.nucleosome_integrated[i][0]
                        one_nucleosome_for_final_nucleosome_temp_recorder[2]=self.nucleosome_integrated[i][2]
                        one_nucleosome_for_final_nucleosome_temp_recorder[6]=self.nucleosome_integrated[i][6]
                        one_nucleosome_for_final_nucleosome_temp_recorder[8]=self.nucleosome_integrated[i][7]
                elif merging_choice_list[i]=='none':
                    if merging_switch=='off':
                        final_nucleosome_temp_recorder.append(self.nucleosome_integrated[i][0:6]+[self.nucleosome_integrated[i][6]]+[self.nucleosome_integrated[i][6]]+[self.nucleosome_integrated[i][7]])    ### 记录这个
                        unmerging_counting=unmerging_counting+1
                    elif merging_switch=='on':
                        final_nucleosome_temp_recorder.append(self.nucleosome_integrated[i-1][0:6]+[self.nucleosome_integrated[i-1][6]]+[self.nucleosome_integrated[i-1][6]]+[self.nucleosome_integrated[i-1][7]])    ### 记录前一个
                        final_nucleosome_temp_recorder.append(self.nucleosome_integrated[i][0:6]+[self.nucleosome_integrated[i][6]]+[self.nucleosome_integrated[i][6]]+[self.nucleosome_integrated[i][7]])    ### 记录这个
                        unmerging_counting=unmerging_counting+2
                        merging_switch='off'
                        one_nucleosome_for_final_nucleosome_temp_recorder=['beginning pairBC', 'ending pairEC', 'beginning pairBI', 'ending pairEI', 'height', 'area', '[beginning peak region]', '[ending peak region]', 'main/shoulder']
                elif merging_choice_list[i]=='left':
                    if merging_switch=='off':
                        final_nucleosome_temp_recorder.append(self.nucleosome_integrated[i][0:6]+[self.nucleosome_integrated[i][6]]+[self.nucleosome_integrated[i][6]]+[self.nucleosome_integrated[i][7]])    ### 记录这个
                        unmerging_counting=unmerging_counting+1
                    elif merging_switch=='on':
                        one_nucleosome_for_final_nucleosome_temp_recorder[1]=self.nucleosome_integrated[i][1]
                        one_nucleosome_for_final_nucleosome_temp_recorder[3]=self.nucleosome_integrated[i][3]
                        one_nucleosome_for_final_nucleosome_temp_recorder[7]=self.nucleosome_integrated[i][6]
                        if type(one_nucleosome_for_final_nucleosome_temp_recorder[0])==int and type(one_nucleosome_for_final_nucleosome_temp_recorder[1])==int and type(one_nucleosome_for_final_nucleosome_temp_recorder[2])==int and type(one_nucleosome_for_final_nucleosome_temp_recorder[3])==int and len(one_nucleosome_for_final_nucleosome_temp_recorder[6])==9 and len(one_nucleosome_for_final_nucleosome_temp_recorder[7])==9:
                            if ('nucleosome' in one_nucleosome_for_final_nucleosome_temp_recorder[8]) or ('nucleosome' in self.nucleosome_integrated[i][7]):
                                one_nucleosome_for_final_nucleosome_temp_recorder[8]='Main_nucleosome:integrated:doublet'
                            else:
                                one_nucleosome_for_final_nucleosome_temp_recorder[8]='Shoulder:doublet'
                            final_nucleosome_temp_recorder.append(one_nucleosome_for_final_nucleosome_temp_recorder)    ### 记录这个
                            merging_counting=merging_counting+1
                            merging_switch='off'
                            one_nucleosome_for_final_nucleosome_temp_recorder=['beginning pairBC', 'ending pairEC', 'beginning pairBI', 'ending pairEI', 'height', 'area', '[beginning peak region]', '[ending peak region]', 'main/shoulder']
                        else:
                            merging_switch='off'
                            one_nucleosome_for_final_nucleosome_temp_recorder=['beginning pairBC', 'ending pairEC', 'beginning pairBI', 'ending pairEI', 'height', 'area', '[beginning peak region]', '[ending peak region]', 'main/shoulder']
                            print('Error in merging!')
            print('\tChecking ==> Number of merged nuclosomes (doublet):  ',merging_counting)
            print('\tChecking ==> Number of unmerged nucleosomes:  ',unmerging_counting)
            print('\tChecking ==> ',merging_counting,'* 2 +',unmerging_counting,'=',merging_counting*2+unmerging_counting)
            print('\tChecking ==> Nucleosomes after doublet merged:  ',len(final_nucleosome_temp_recorder))
            print('\tChecking ==> ',merging_counting,'+',unmerging_counting,'=',merging_counting+unmerging_counting)
            ################################
            final_nucleosome_temp_recorder_noise_discarded=[]    ### 记录器
            discarded_noise=0
            for i in range(len(final_nucleosome_temp_recorder)):
                original_score_fragment=[]
                LoG_fragment=[]
                negative_counting=0
                negative_counting_switch='off'
                negative_counting_list=[]
                LoG_Sigma3_length=0
                for nnp in range(final_nucleosome_temp_recorder[i][2],final_nucleosome_temp_recorder[i][3]+1):
                    original_score_fragment.append(self.score_table[nnp][1])
                    LoG_fragment.append(self.score_table[nnp][3])
                    if self.score_table[nnp][3]<0:
                        if nnp<final_nucleosome_temp_recorder[i][3]:
                            negative_counting_switch='on'
                            negative_counting=negative_counting+1
                        elif nnp==final_nucleosome_temp_recorder[i][3]:
                            negative_counting=negative_counting+1
                            negative_counting_list.append(negative_counting)
                            negative_counting=0
                            negative_counting_switch='off'
                    elif self.score_table[nnp][3]>=0:
                        if negative_counting_switch=='on':
                            negative_counting_list.append(negative_counting)
                            negative_counting=0
                            negative_counting_switch='off'
                        elif negative_counting_switch=='off':
                            pass
                LoG_Sigma3_length=max(negative_counting_list)
                if sum(original_score_fragment)<self.threshold['discarded_noise_selfAUC_1'] and LoG_Sigma3_length<=self.threshold['discarded_noise_selfLength_1'] and sum(LoG_fragment)/len(LoG_fragment)>self.threshold['discarded_noise_LoG_average_1']:
                    discarded_noise=discarded_noise+1
                elif len(original_score_fragment)<=self.threshold['discarded_noise_selfLength_2'] and sum(LoG_fragment)/len(LoG_fragment)>self.threshold['discarded_noise_LoG_average_2']:
                    discarded_noise=discarded_noise+1
                elif max(original_score_fragment)<=self.threshold['discarded_noise_selfHeight_3'] and sum(LoG_fragment)/len(LoG_fragment)>self.threshold['discarded_noise_LoG_average_3']:
                    discarded_noise=discarded_noise+1
                elif LoG_Sigma3_length<=self.threshold['discarded_noise_RealSelfLength_4'] and sum(LoG_fragment)/len(LoG_fragment)>self.threshold['discarded_noise_LoG_average_4']:
                    discarded_noise=discarded_noise+1
                elif LoG_Sigma3_length<=self.threshold['discarded_noise_RealSelfLength_5'] and sum(LoG_fragment)/len(LoG_fragment)>self.threshold['discarded_noise_LoG_average_5']:
                    discarded_noise=discarded_noise+1
                elif LoG_Sigma3_length<=self.threshold['discarded_noise_RealSelfLength_6'] and sum(LoG_fragment)/len(LoG_fragment)>self.threshold['discarded_noise_LoG_average_6']:
                    discarded_noise=discarded_noise+1
                else:
                    final_nucleosome_temp_recorder_noise_discarded.append(final_nucleosome_temp_recorder[i])
            print('\tChecking ==> Discarded nucleosomes as noise:  ',discarded_noise)
            print('\tChecking ==> Final remaining nucleosomes:  ',len(final_nucleosome_temp_recorder_noise_discarded))
            return final_nucleosome_temp_recorder_noise_discarded

        if threshold['filter_switch']=='off':
            ######## 重要记录器 ###################################
            self.final_nucleosome=[]
            ######## 重要记录器 ###################################
            for i in range(len(self.nucleosome_integrated)):    ### 新的记录
                self.final_nucleosome.append(self.nucleosome_integrated[i][0:6]+[self.nucleosome_integrated[i][6]]+[self.nucleosome_integrated[i][6]]+[self.nucleosome_integrated[i][7]])
        elif threshold['filter_switch']=='on':
            ######## 重要记录器 ###################################
            self.final_nucleosome=[]
            ######## 重要记录器 ###################################
            self.final_nucleosome=merge_and_annotate(self,interval_property_list)
        print('\tChecking ==> Nucleosomes after filtered:  ',len(self.final_nucleosome))
        print('...... Results are filtered and annotated.','\t',time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))
            


    def record_results(self):
        print('Record the detected nucleosomes in like-wiggle table ......')
        record_counting=0
        for j in range(len(self.final_nucleosome)):
            original_score_fragment=[]
            for nnp in range(self.final_nucleosome[j][2],self.final_nucleosome[j][3]+1):
                original_score_fragment.append(self.score_table[nnp][1])
            if len(original_score_fragment)==0:
                pass
            else:
                self.final_nucleosome[j][4]=max(original_score_fragment)
                self.final_nucleosome[j][5]=sum(original_score_fragment)
                for nnp in range(self.final_nucleosome[j][2],self.final_nucleosome[j][3]+1):
                    self.score_table[nnp][4]=max(original_score_fragment)    ### 在self.score_table里做记录
                record_counting=record_counting+1
        print('\tChecking ==> recorded nucleosomes:  ',record_counting)
        print('...... Nucleosomes are recorded in like-wiggle table.','\t',time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))

    def Test_significance( self ):
        ##self.Pvalue_Peak_dict = { }
        self.Log_Pvalue_Peak_dict = { }
        ##self.Pvalue_LeftValley_dict = { }
        ##self.Pvalue_RightValley_dict = { }
        self.Log_Pvalue_Valley_dict = { }
        ########
        print('Test significance ......')
        next_done = 'no'
        beginning_coordinate = 1
        ending_coordinate = len( self.tag_list ) * 10 - 9
        for i in range( len( self.final_nucleosome ) ):
            # sys.stdout.write( '\r    Nucleosome: '+str(i+1) )
            ### nucleosome=[[pairBC, pairEC, pairBI, pairEI, 'height', 'area', [peakBC,maxBC,maxEC,peakEC,peakBI,maxBI,maxEI,peakEI,BeginningPeakIndex], [peakBC,maxBC,maxEC,peakEC,peakBI,maxBI,maxEI,peakEI,EndingPeakIndex], main/shoulder], ...]
            LeftInflection  = self.final_nucleosome[i][0]    #### 峰的起点坐标
            RightInflection = self.final_nucleosome[i][1]    #### 峰的终点坐标
            Nucleosome_mid = ( LeftInflection + RightInflection ) // 2 // 10 * 10 + 1
            Nuc_Background_left_1000  = max( beginning_coordinate  , Nucleosome_mid - 1000 )
            Nuc_Background_right_1000 = min( Nucleosome_mid + 1000 , ending_coordinate     )
            Nuc_Background_left_5000  = max( beginning_coordinate  , Nucleosome_mid - 5000 )
            Nuc_Background_right_5000 = min( Nucleosome_mid + 5000 , ending_coordinate     )
            Nuc_Background_left_10000  = max( beginning_coordinate   , Nucleosome_mid - 10000 )
            Nuc_Background_right_10000 = min( Nucleosome_mid + 10000 , ending_coordinate      )
            ####
            LeftInflection_index  = ( LeftInflection - beginning_coordinate ) // 10
            RightInflection_index = ( RightInflection - beginning_coordinate ) // 10
            Nuc_Background_left_1000_index  = ( Nuc_Background_left_1000 - beginning_coordinate ) // 10
            Nuc_Background_right_1000_index = ( Nuc_Background_right_1000 - beginning_coordinate ) // 10
            Nuc_Background_left_5000_index  = ( Nuc_Background_left_5000 - beginning_coordinate ) // 10
            Nuc_Background_right_5000_index = ( Nuc_Background_right_5000 - beginning_coordinate ) // 10
            Nuc_Background_left_10000_index  = ( Nuc_Background_left_10000 - beginning_coordinate ) // 10
            Nuc_Background_right_10000_index = ( Nuc_Background_right_10000 - beginning_coordinate ) // 10
            Background_1000_for_peak  = sum( [ self.tag_list[index] for index in range( Nuc_Background_left_1000_index , Nuc_Background_right_1000_index+1 ) ] ) / (Nuc_Background_right_1000_index-Nuc_Background_left_1000_index+1) * (RightInflection_index-LeftInflection_index+1)
            Background_5000_for_peak  = sum( [ self.tag_list[index] for index in range( Nuc_Background_left_5000_index , Nuc_Background_right_5000_index+1 ) ] ) / (Nuc_Background_right_5000_index-Nuc_Background_left_5000_index+1) * (RightInflection_index-LeftInflection_index+1)
            Background_10000_for_peak = sum( [ self.tag_list[index] for index in range( Nuc_Background_left_10000_index , Nuc_Background_right_10000_index+1 ) ] ) / (Nuc_Background_right_10000_index-Nuc_Background_left_10000_index+1) * (RightInflection_index-LeftInflection_index+1)
            Background_for_peak = max( Background_1000_for_peak , Background_5000_for_peak , Background_10000_for_peak )
            foreground_for_peak = sum( [ self.tag_list[index] for index in range( LeftInflection_index , RightInflection_index+1 ) ] )
            Pvalue_for_peak , Score_for_peak = Poisson_test.greater_fast( Background_for_peak , foreground_for_peak )
            ########
            if i == 0:
                LeftValley_end = self.final_nucleosome[i][0] - 10    #### 左侧谷的终点坐标
                LeftValley_start = max( LeftValley_end - 300 , beginning_coordinate )    #### 左侧谷的起点坐标
                LeftValley_mid = ( LeftValley_start + LeftValley_end ) // 2 // 10 * 10 + 1
                LV_Background_left_1000  = max( beginning_coordinate  , LeftValley_mid - 1000 )
                LV_Background_right_1000 = min( LeftValley_mid + 1000 , ending_coordinate     )
                LV_Background_left_5000  = max( beginning_coordinate  , LeftValley_mid - 5000 )
                LV_Background_right_5000 = min( LeftValley_mid + 5000 , ending_coordinate     )
                LV_Background_left_10000  = max( beginning_coordinate  , LeftValley_mid - 10000 )
                LV_Background_right_10000 = min( LeftValley_mid + 10000 , ending_coordinate     )
                ####
                LeftValley_start_index = ( LeftValley_start - beginning_coordinate ) // 10
                LeftValley_end_index   = ( LeftValley_end - beginning_coordinate ) // 10
                LV_Background_left_1000_index  = ( LV_Background_left_1000 - beginning_coordinate ) // 10
                LV_Background_right_1000_index = ( LV_Background_right_1000 - beginning_coordinate ) // 10
                LV_Background_left_5000_index  = ( LV_Background_left_5000 - beginning_coordinate ) // 10
                LV_Background_right_5000_index = ( LV_Background_right_5000 - beginning_coordinate ) // 10
                LV_Background_left_10000_index  = ( LV_Background_left_10000 - beginning_coordinate ) // 10
                LV_Background_right_10000_index = ( LV_Background_right_10000 - beginning_coordinate ) // 10
                Background_1000_for_LV  = sum( [ self.tag_list[index] for index in range( LV_Background_left_1000_index , LV_Background_right_1000_index+1 ) ] ) / (LV_Background_right_1000_index-LV_Background_left_1000_index+1) * (LeftValley_end_index-LeftValley_start_index+1)
                Background_5000_for_LV  = sum( [ self.tag_list[index] for index in range( LV_Background_left_5000_index , LV_Background_right_5000_index+1 ) ] ) / (LV_Background_right_5000_index-LV_Background_left_5000_index+1) * (LeftValley_end_index-LeftValley_start_index+1)
                Background_10000_for_LV = sum( [ self.tag_list[index] for index in range( LV_Background_left_10000_index , LV_Background_right_10000_index+1 ) ] ) / (LV_Background_right_10000_index-LV_Background_left_10000_index+1) * (LeftValley_end_index-LeftValley_start_index+1)
                Background_for_LV = min( Background_1000_for_LV , Background_5000_for_LV , Background_10000_for_LV )
                if Background_for_LV < 0:
                    Background_for_LV = 0
                foreground_for_LV = sum( [ self.tag_list[index] for index in range( LeftValley_start_index , LeftValley_end_index+1 ) ] )
                Pvalue_for_LV , Score_for_LV = Poisson_test.less_fast( Background_for_LV , foreground_for_LV )
                ########
                RightValley_start = self.final_nucleosome[i][1] + 10    #### 右侧谷的起点坐标
                if len( self.final_nucleosome ) == 1:
                    RightValley_end = min( RightValley_start + 300 , ending_coordinate )    #### 右侧谷的终点坐标
                elif len( self.final_nucleosome ) > 1:
                    if self.final_nucleosome[i+1][0] - RightValley_start <= 1000:
                        RightValley_end = self.final_nucleosome[i+1][0] - 10    #### 右侧谷的终点坐标
                        next_done = 'yes'
                    else:
                        RightValley_end = RightValley_start + 1000    #### 右侧谷的终点坐标
                        next_done = 'no'
                RightValley_mid = ( RightValley_start + RightValley_end ) // 2 // 10 * 10 + 1
                RV_Background_left_1000  = max( beginning_coordinate   , RightValley_mid - 1000 )
                RV_Background_right_1000 = min( RightValley_mid + 1000 , ending_coordinate      )
                RV_Background_left_5000  = max( beginning_coordinate   , RightValley_mid - 5000 )
                RV_Background_right_5000 = min( RightValley_mid + 5000 , ending_coordinate      )
                RV_Background_left_10000  = max( beginning_coordinate   , RightValley_mid - 10000 )
                RV_Background_right_10000 = min( RightValley_mid + 10000 , ending_coordinate      )
                ####
                RightValley_start_index = ( RightValley_start - beginning_coordinate ) // 10
                RightValley_end_index   = ( RightValley_end - beginning_coordinate ) // 10
                RV_Background_left_1000_index  = ( RV_Background_left_1000 - beginning_coordinate ) // 10
                RV_Background_right_1000_index = ( RV_Background_right_1000 - beginning_coordinate ) // 10
                RV_Background_left_5000_index  = ( RV_Background_left_5000 - beginning_coordinate ) // 10
                RV_Background_right_5000_index = ( RV_Background_right_5000 - beginning_coordinate ) // 10
                RV_Background_left_10000_index  = ( RV_Background_left_10000 - beginning_coordinate ) // 10
                RV_Background_right_10000_index = ( RV_Background_right_10000 - beginning_coordinate ) // 10
                Background_1000_for_RV  = sum( [ self.tag_list[index] for index in range( RV_Background_left_1000_index , RV_Background_right_1000_index+1 ) ] ) / (RV_Background_right_1000_index-RV_Background_left_1000_index+1) * (RightValley_end_index-RightValley_start_index+1)
                Background_5000_for_RV  = sum( [ self.tag_list[index] for index in range( RV_Background_left_5000_index , RV_Background_right_5000_index+1 ) ] ) / (RV_Background_right_5000_index-RV_Background_left_5000_index+1) * (RightValley_end_index-RightValley_start_index+1)
                Background_10000_for_RV = sum( [ self.tag_list[index] for index in range( RV_Background_left_10000_index , RV_Background_right_10000_index+1 ) ] ) / (RV_Background_right_10000_index-RV_Background_left_10000_index+1) * (RightValley_end_index-RightValley_start_index+1)
                Background_for_RV = min( Background_1000_for_RV , Background_5000_for_RV , Background_10000_for_RV )
                if Background_for_RV < 0:
                    Background_for_RV = 0
                foreground_for_RV = sum( [ self.tag_list[index] for index in range( RightValley_start_index , RightValley_end_index+1 ) ] )
                Pvalue_for_RV , Score_for_RV = Poisson_test.less_fast( Background_for_RV , foreground_for_RV )
            elif ( i > 0 ) and ( i < len( self.final_nucleosome ) - 1 ):
                if next_done == 'yes':
                    Pvalue_for_LV , Score_for_LV = Pvalue_for_RV , Score_for_RV
                    next_done = 'no'
                elif next_done == 'no':
                    LeftValley_end = self.final_nucleosome[i][0] - 10    #### 左侧谷的终点坐标
                    LeftValley_start = max( LeftValley_end - 1000 , beginning_coordinate )    #### 左侧谷的起点坐标
                    LeftValley_mid = ( LeftValley_start + LeftValley_end ) // 2 // 10 * 10 + 1
                    LV_Background_left_1000  = max( beginning_coordinate  , LeftValley_mid - 1000 )
                    LV_Background_right_1000 = min( LeftValley_mid + 1000 , ending_coordinate     )
                    LV_Background_left_5000  = max( beginning_coordinate  , LeftValley_mid - 5000 )
                    LV_Background_right_5000 = min( LeftValley_mid + 5000 , ending_coordinate     )
                    LV_Background_left_10000  = max( beginning_coordinate  , LeftValley_mid - 10000 )
                    LV_Background_right_10000 = min( LeftValley_mid + 10000 , ending_coordinate     )
                    ####
                    LeftValley_start_index = ( LeftValley_start - beginning_coordinate ) // 10
                    LeftValley_end_index   = ( LeftValley_end - beginning_coordinate ) // 10
                    LV_Background_left_1000_index  = ( LV_Background_left_1000 - beginning_coordinate ) // 10
                    LV_Background_right_1000_index = ( LV_Background_right_1000 - beginning_coordinate ) // 10
                    LV_Background_left_5000_index  = ( LV_Background_left_5000 - beginning_coordinate ) // 10
                    LV_Background_right_5000_index = ( LV_Background_right_5000 - beginning_coordinate ) // 10
                    LV_Background_left_10000_index  = ( LV_Background_left_10000 - beginning_coordinate ) // 10
                    LV_Background_right_10000_index = ( LV_Background_right_10000 - beginning_coordinate ) // 10
                    Background_1000_for_LV  = sum( [ self.tag_list[index] for index in range( LV_Background_left_1000_index , LV_Background_right_1000_index+1 ) ] ) / (LV_Background_right_1000_index-LV_Background_left_1000_index+1) * (LeftValley_end_index-LeftValley_start_index+1)
                    Background_5000_for_LV  = sum( [ self.tag_list[index] for index in range( LV_Background_left_5000_index , LV_Background_right_5000_index+1 ) ] ) / (LV_Background_right_5000_index-LV_Background_left_5000_index+1) * (LeftValley_end_index-LeftValley_start_index+1)
                    Background_10000_for_LV = sum( [ self.tag_list[index] for index in range( LV_Background_left_10000_index , LV_Background_right_10000_index+1 ) ] ) / (LV_Background_right_10000_index-LV_Background_left_10000_index+1) * (LeftValley_end_index-LeftValley_start_index+1)
                    Background_for_LV = min( Background_1000_for_LV , Background_5000_for_LV , Background_10000_for_LV )
                    if Background_for_LV < 0:
                        Background_for_LV = 0
                    foreground_for_LV = sum( [ self.tag_list[index] for index in range( LeftValley_start_index , LeftValley_end_index+1 ) ] )
                    Pvalue_for_LV , Score_for_LV = Poisson_test.less_fast( Background_for_LV , foreground_for_LV )
                ########
                RightValley_start = self.final_nucleosome[i][1] + 10    #### 右侧谷的起点坐标
                if self.final_nucleosome[i+1][0] - RightValley_start <= 1000:
                    RightValley_end = self.final_nucleosome[i+1][0] - 10    #### 右侧谷的终点坐标
                    next_done = 'yes'
                else:
                    RightValley_end = RightValley_start + 1000    #### 右侧谷的终点坐标
                    next_done = 'no'
                RightValley_mid = ( RightValley_start + RightValley_end ) // 2 // 10 * 10 + 1
                RV_Background_left_1000  = max( beginning_coordinate   , RightValley_mid - 1000 )
                RV_Background_right_1000 = min( RightValley_mid + 1000 , ending_coordinate      )
                RV_Background_left_5000  = max( beginning_coordinate   , RightValley_mid - 5000 )
                RV_Background_right_5000 = min( RightValley_mid + 5000 , ending_coordinate      )
                RV_Background_left_10000  = max( beginning_coordinate   , RightValley_mid - 10000 )
                RV_Background_right_10000 = min( RightValley_mid + 10000 , ending_coordinate      )
                ####
                RightValley_start_index = ( RightValley_start - beginning_coordinate ) // 10
                RightValley_end_index   = ( RightValley_end - beginning_coordinate ) // 10
                RV_Background_left_1000_index  = ( RV_Background_left_1000 - beginning_coordinate ) // 10
                RV_Background_right_1000_index = ( RV_Background_right_1000 - beginning_coordinate ) // 10
                RV_Background_left_5000_index  = ( RV_Background_left_5000 - beginning_coordinate ) // 10
                RV_Background_right_5000_index = ( RV_Background_right_5000 - beginning_coordinate ) // 10
                RV_Background_left_10000_index  = ( RV_Background_left_10000 - beginning_coordinate ) // 10
                RV_Background_right_10000_index = ( RV_Background_right_10000 - beginning_coordinate ) // 10
                Background_1000_for_RV  = sum( [ self.tag_list[index] for index in range( RV_Background_left_1000_index , RV_Background_right_1000_index+1 ) ] ) / (RV_Background_right_1000_index-RV_Background_left_1000_index+1) * (RightValley_end_index-RightValley_start_index+1)
                Background_5000_for_RV  = sum( [ self.tag_list[index] for index in range( RV_Background_left_5000_index , RV_Background_right_5000_index+1 ) ] ) / (RV_Background_right_5000_index-RV_Background_left_5000_index+1) * (RightValley_end_index-RightValley_start_index+1)
                Background_10000_for_RV = sum( [ self.tag_list[index] for index in range( RV_Background_left_10000_index , RV_Background_right_10000_index+1 ) ] ) / (RV_Background_right_10000_index-RV_Background_left_10000_index+1) * (RightValley_end_index-RightValley_start_index+1)
                Background_for_RV = min( Background_1000_for_RV , Background_5000_for_RV , Background_10000_for_RV )
                if Background_for_RV < 0:
                    Background_for_RV = 0
                foreground_for_RV = sum( [ self.tag_list[index] for index in range( RightValley_start_index , RightValley_end_index+1 ) ] )
                Pvalue_for_RV , Score_for_RV = Poisson_test.less_fast( Background_for_RV , foreground_for_RV )
            elif i == len( self.final_nucleosome ) - 1:
                if next_done == 'yes':
                    Pvalue_for_LV , Score_for_LV = Pvalue_for_RV , Score_for_RV
                    next_done = 'no'
                elif next_done == 'no':
                    LeftValley_end = self.final_nucleosome[i][0] - 10    #### 左侧谷的终点坐标
                    LeftValley_start = max( LeftValley_end - 1000 , beginning_coordinate )    #### 左侧谷的起点坐标
                    LeftValley_mid = ( LeftValley_start + LeftValley_end ) // 2 // 10 * 10 + 1
                    LV_Background_left_1000  = max( beginning_coordinate  , LeftValley_mid - 1000 )
                    LV_Background_right_1000 = min( LeftValley_mid + 1000 , ending_coordinate     )
                    LV_Background_left_5000  = max( beginning_coordinate  , LeftValley_mid - 5000 )
                    LV_Background_right_5000 = min( LeftValley_mid + 5000 , ending_coordinate     )
                    LV_Background_left_10000  = max( beginning_coordinate  , LeftValley_mid - 10000 )
                    LV_Background_right_10000 = min( LeftValley_mid + 10000 , ending_coordinate     )
                    ####
                    LeftValley_start_index = ( LeftValley_start - beginning_coordinate ) // 10
                    LeftValley_end_index   = ( LeftValley_end - beginning_coordinate ) // 10
                    LV_Background_left_1000_index  = ( LV_Background_left_1000 - beginning_coordinate ) // 10
                    LV_Background_right_1000_index = ( LV_Background_right_1000 - beginning_coordinate ) // 10
                    LV_Background_left_5000_index  = ( LV_Background_left_5000 - beginning_coordinate ) // 10
                    LV_Background_right_5000_index = ( LV_Background_right_5000 - beginning_coordinate ) // 10
                    LV_Background_left_10000_index  = ( LV_Background_left_10000 - beginning_coordinate ) // 10
                    LV_Background_right_10000_index = ( LV_Background_right_10000 - beginning_coordinate ) // 10
                    Background_1000_for_LV  = sum( [ self.tag_list[index] for index in range( LV_Background_left_1000_index , LV_Background_right_1000_index+1 ) ] ) / (LV_Background_right_1000_index-LV_Background_left_1000_index+1) * (LeftValley_end_index-LeftValley_start_index+1)
                    Background_5000_for_LV  = sum( [ self.tag_list[index] for index in range( LV_Background_left_5000_index , LV_Background_right_5000_index+1 ) ] ) / (LV_Background_right_5000_index-LV_Background_left_5000_index+1) * (LeftValley_end_index-LeftValley_start_index+1)
                    Background_10000_for_LV = sum( [ self.tag_list[index] for index in range( LV_Background_left_10000_index , LV_Background_right_10000_index+1 ) ] ) / (LV_Background_right_10000_index-LV_Background_left_10000_index+1) * (LeftValley_end_index-LeftValley_start_index+1)
                    Background_for_LV = min( Background_1000_for_LV , Background_5000_for_LV , Background_10000_for_LV )
                    if Background_for_LV < 0:
                        Background_for_LV = 0
                    foreground_for_LV = sum( [ self.tag_list[index] for index in range( LeftValley_start_index , LeftValley_end_index+1 ) ] )
                    Pvalue_for_LV , Score_for_LV = Poisson_test.less_fast( Background_for_LV , foreground_for_LV )
                ########
                RightValley_start = self.final_nucleosome[i][1] + 10    #### 右侧谷的起点坐标
                RightValley_end = min( RightValley_start + 300 , ending_coordinate )    #### 右侧谷的终点坐标
                RightValley_mid = ( RightValley_start + RightValley_end ) // 2 // 10 * 10 + 1
                RV_Background_left_1000  = max( beginning_coordinate   , RightValley_mid - 1000 )
                RV_Background_right_1000 = min( RightValley_mid + 1000 , ending_coordinate      )
                RV_Background_left_5000  = max( beginning_coordinate   , RightValley_mid - 5000 )
                RV_Background_right_5000 = min( RightValley_mid + 5000 , ending_coordinate      )
                RV_Background_left_10000  = max( beginning_coordinate   , RightValley_mid - 10000 )
                RV_Background_right_10000 = min( RightValley_mid + 10000 , ending_coordinate      )
                ####
                RightValley_start_index = ( RightValley_start - beginning_coordinate ) // 10
                RightValley_end_index   = ( RightValley_end - beginning_coordinate ) // 10
                RV_Background_left_1000_index  = ( RV_Background_left_1000 - beginning_coordinate ) // 10
                RV_Background_right_1000_index = ( RV_Background_right_1000 - beginning_coordinate ) // 10
                RV_Background_left_5000_index  = ( RV_Background_left_5000 - beginning_coordinate ) // 10
                RV_Background_right_5000_index = ( RV_Background_right_5000 - beginning_coordinate ) // 10
                RV_Background_left_10000_index  = ( RV_Background_left_10000 - beginning_coordinate ) // 10
                RV_Background_right_10000_index = ( RV_Background_right_10000 - beginning_coordinate ) // 10
                Background_1000_for_RV  = sum( [ self.tag_list[index] for index in range( RV_Background_left_1000_index , RV_Background_right_1000_index+1 ) ] ) / (RV_Background_right_1000_index-RV_Background_left_1000_index+1) * (RightValley_end_index-RightValley_start_index+1)
                Background_5000_for_RV  = sum( [ self.tag_list[index] for index in range( RV_Background_left_5000_index , RV_Background_right_5000_index+1 ) ] ) / (RV_Background_right_5000_index-RV_Background_left_5000_index+1) * (RightValley_end_index-RightValley_start_index+1)
                Background_10000_for_RV = sum( [ self.tag_list[index] for index in range( RV_Background_left_10000_index , RV_Background_right_10000_index+1 ) ] ) / (RV_Background_right_10000_index-RV_Background_left_10000_index+1) * (RightValley_end_index-RightValley_start_index+1)
                Background_for_RV = min( Background_1000_for_RV , Background_5000_for_RV , Background_10000_for_RV )
                if Background_for_RV < 0:
                    Background_for_RV = 0
                foreground_for_RV = sum( [ self.tag_list[index] for index in range( RightValley_start_index , RightValley_end_index+1 ) ] )
                Pvalue_for_RV , Score_for_RV = Poisson_test.less_fast( Background_for_RV , foreground_for_RV )
            else:
                print('\tError ==> Nucleosome: ' , i )
            Score_for_V = ( Score_for_LV + Score_for_RV ) * 0.5
            ##self.Pvalue_Peak_dict[i]        = Pvalue_for_peak
            ##self.Pvalue_LeftValley_dict[i]  = Pvalue_for_LV
            ##self.Pvalue_RightValley_dict[i] = Pvalue_for_RV
            self.Log_Pvalue_Peak_dict[i]   = Score_for_peak
            self.Log_Pvalue_Valley_dict[i] = Score_for_V
            # sys.stdout.flush()
            # print('\tChecking ==> Nucleosomes: ', i+1 )
        print('\tChecking ==> self.Log_Pvalue_Peak_dict:   ' , len( self.Log_Pvalue_Peak_dict )   )
        print('\tChecking ==> self.Log_Pvalue_Valley_dict: ' , len( self.Log_Pvalue_Valley_dict ) )
        print('...... Statistic tests are finished.','\t',time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))
            

    def write_wiggle_file(self):
        print('Generating like-wiggle file ......')
        wiggle=open(self.outputfile_wiggle,'w')
        wiggle.write('track type=like_wiggle_by_CHEN,Weizhong\nvariableStep chromosome='+self.chromosome+' span=10\n')
        ####wiggle.write('1:coordinate\t2:original_scoring\t3:Gaussian_convolution_smoothing\t4:First_Derivative_of_Gaussian_convolution\t5:LoG\t6:Tirst_Derivative_of_Gaussian_convolution\t7:Nucloesome_determination\n')
        wiggle.write('1:coordinate\t2:original_scoring\t3:Gaussian_convolution_smoothing\t4:LoG\t5:Adjusted_LoG\t6:Nucloesome_determination\n')
        line=0
        for k in range(len(self.score_list)):
            ####wiggle.write(str(k*10+1)+'\t'+str(self.score_list[k])+'\t'+str(self.Gaussian_list[k])+'\t'+str(self.FDoG_list[k])+'\t'+str(self.LoG_list[k])+'\t'+str(self.TDoG_list[k])+'\t'+str(self.secondary_LoG_list[k])+'\t'+str(score_file[k][4])+'\n')
            wiggle.write(str(k*10+1)+'\t'+str(self.score_list[k])+'\t'+str(self.Gaussian_list[k])+'\t'+str(self.LoG_list[k])+'\t'+str(self.secondary_LoG_list[k])+'\t'+str(self.score_table[k][4])+'\n')
            line=line+1
        wiggle.close()
        print('...... Wiggle file is finished.','\t',time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))
        print('\tChecking ==> Total line of the wiggle file:  ',line)

    def write_nucleosome_file(self):
        print('Generating nucleosome collection file ......')
        nucleosome_beginning_ending_file=open(self.outputfile_nucleosome,'w')
        nucleosome_beginning_ending_file.write('chromosome='+self.chromosome+'  total_nucleosome:'+str(len(self.final_nucleosome))+'  coordinate:hg18\n')
        nucleosome_beginning_ending_file.write('1:Chromosome\t2:Inflection_Pair_Beginning\t3:Inflection_Pair_Ending\t4:Length_between_Inflection\t5:Height_of_nucleosome\t6:Area_under_Curve\t7:Nucleosome_Index\t8:Beginning_Peak_Region\t9:Ending_Peak_Region\t10:Peak_Region_Length\t11:Physical_Property\n')
        line=0
        for k in range(len(self.final_nucleosome)):
            line=line+1
            ### nucleosome=[[pairBC, pairEC, pairBI, pairEI, 'height', 'area', [peakBC,maxBC,maxEC,peakEC,peakBI,maxBI,maxEI,peakEI,BeginningPeakIndex], [peakBC,maxBC,maxEC,peakEC,peakBI,maxBI,maxEI,peakEI,EndingPeakIndex], main/shoulder], ...]
            nucleosome_beginning_ending_file.write(self.chromosome+'\t')    ### 1:Chromosome
            nucleosome_beginning_ending_file.write(str(self.final_nucleosome[k][0])+'\t')    ### 2:Inflection_Pair_Beginning
            nucleosome_beginning_ending_file.write(str(self.final_nucleosome[k][1])+'\t')    ### 3:Inflection_Pair_Ending
            nucleosome_beginning_ending_file.write(str(self.final_nucleosome[k][1]-self.final_nucleosome[k][0]+10)+'\t')    ### 4:Length_between_Inflection
            nucleosome_beginning_ending_file.write(str(self.final_nucleosome[k][4])+'\t')    ### 5:Height_of_Nucleosome
            nucleosome_beginning_ending_file.write(str(self.final_nucleosome[k][5])+'\t')    ### 6:Area_under_Curve
            nucleosome_beginning_ending_file.write('Nucleosome:'+str(line)+'\t')    ### 7:Nucleosome_Index
            nucleosome_beginning_ending_file.write('BeginningPeak:'+str(self.final_nucleosome[k][6][8]))    ### 8:Beginning_Peak_Region
            nucleosome_beginning_ending_file.write('('+str(self.final_nucleosome[k][6][0])+'--'+str(self.final_nucleosome[k][6][1])+'--'+str(self.final_nucleosome[k][6][2])+'--'+str(self.final_nucleosome[k][6][3])+')'+'\t')
            nucleosome_beginning_ending_file.write('EndingPeak:'+str(self.final_nucleosome[k][7][8]))    ### 9:Ending_Peak_Region
            nucleosome_beginning_ending_file.write('('+str(self.final_nucleosome[k][7][0])+'--'+str(self.final_nucleosome[k][7][1])+'--'+str(self.final_nucleosome[k][7][2])+'--'+str(self.final_nucleosome[k][7][3])+')'+'\t')
            nucleosome_beginning_ending_file.write(str(self.final_nucleosome[k][7][3]-self.final_nucleosome[k][6][0]+10)+'\t')    ### 10:Peak_Region_Length
            nucleosome_beginning_ending_file.write(str(self.final_nucleosome[k][8])+'\n')    ### 11:Physical_Property
        nucleosome_beginning_ending_file.close()
        print('...... Nucleosome collection file is finished.','\t',time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))
        print('\tChecking ==> Total line of the collection file:  ',line)



class NucleosomeAccuratePositioning(MainProgram):
    def __init__(self,FilesParameters,ConvolutionParameters,threshold):
        print(time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))
        self.inputfile = FilesParameters['inputfile_for_nucleosome_positioning']
        self.chromosome = FilesParameters['Chosen_Chromosome_Abbreviation']
        self.chrlength = FilesParameters['Chosen_Chromosome_Length']
        self.chrRecordingListLength = 0
        self.outputfile_wiggle = FilesParameters['outputfile_like_wig']
        self.outputfile_nucleosome = FilesParameters['outputfile_like_bed']
        print('Input file:  ',self.inputfile)
        ####print('Cell type:   ',self.celltype+' CD4+ T cells from human')
        print('Chromosome:  ',self.chromosome)
        print('Length of',self.chromosome,':  ',self.chrlength)
        print('Output wiggle file of nucleosome distribution: ',self.outputfile_wiggle)
        print('Output file of nucleosome collection:          ',self.outputfile_nucleosome)
        self.ConvolutionParameters = ConvolutionParameters
        self.threshold = threshold
        self.score_list = []
        self.tag_list = [ ]
        self.Gaussian_list = []
        self.FDoG_list = []
        self.LoG_list = []
        self.TDoG_list = []
        self.score_table = []    ### coordinate, self.score_list, self.Gaussian_list, self.LoG_list, column for recording
        ##self.secondary_Gaussian_list = []
        ##self.secondary_FDoG_list = []
        self.secondary_LoG_list = []
        ##self.secondary_TDoG_list = []
        print('======> ======> ======> ======> ======> ======>')
        #MainProgram.__init__(self,FilesParameters,ConvolutionParameters,threshold,chromosome_length)
        ######## 建立对象 self.inputfile
        ######## 建立对象 self.celltype
        ######## 建立对象 self.chromosome
        ######## 建立对象 self.chrlength
        ######## 建立对象 self.outputfile_path
        ######## 建立对象 self.outputfile_wiggle
        ######## 建立对象 self.outputfile_nucleosome
        ######## 建立对象 self.ConvolutionParameters=ConvolutionParameters
        ######## 建立对象 self.threshold=threshold
        ######## 建立对象 self.score_list=[]
        ######## 建立对象 self.Gaussian_list=[]
        ######## 建立对象 self.FDoG_list=[]
        ######## 建立对象 self.LoG_list=[]
        ######## 建立对象 self.TDoG_list=[]
        ######## 建立对象 self.score_table=[]
        ######## 建立对象 self.secondary_Gaussian_list=[]
        ######## 建立对象 self.secondary_FDoG_list=[]
        ######## 建立对象 self.secondary_LoG_list=[]
        ######## 建立对象 self.secondary_TDoG_list=[]

        if FilesParameters[ 'single_or_paired' ] == 's':
            MainProgram.score(self)
        elif FilesParameters[ 'single_or_paired' ] == 'p':
            MainProgram.score_for_Paired_end(self)
        
        MainProgram.Gaussian_convolution_smoothing(self)
        
        MainProgram.extremum_detection(self)
        ######## 建立对象 self.peak_dict={}
        ######## 建立对象 self.peak_details_dict={}
        
        MainProgram.inflection_pairs_detection(self)
        ######## 建立对象 self.inflection_pairs_list=[]
        
        MainProgram.inflection_pairs_midpoint_detection(self)
        ######## 建立对象 self.inflection_pairs_list_with_midpoint=[]
        ######## 建立对象 self.inflection_pairs_dict_with_midpoint={}
        
        MainProgram.secondary_inflection_pairs_detection(self)
        ######## 建立对象 self.secondary_inflection_pairs_list=[]
        ######## 建立对象 self.secondary_inflection_pairs_dict_with_peak={}
        
        MainProgram.preliminary_nucleosome_position(self)
        ######## 建立对象 self.unintegrated_inflection_pairs_list=[]
        ######## 建立对象 self.nucleosome_list=[]
        ######## 建立对象 self.nucleosome_dict={}
        ######## 建立对象 self.shoulder_list=[]
        ######## 建立对象 self.shoulder_dict={}
        ######## 更新 self.peak_dict

        MainProgram.collect_and_sort_shoulders(self)
	######## 建立对象 self.shoulders_sorted_with_main_nucleosomes={}
        
        MainProgram.precise_nucleosome_position(self)
        ######## 建立对象 self.more_independent_shoulders=[]
        ######## 修改 self.nucleosome_dict
        ######## 建立对象 self.nucleosome_integrated=[]
        ######## 建立对象 self.final_nucleosome=[]  ########  self.final_nucleosome <- self.nucleosome_integrated

        MainProgram.adjust_border(self)
        ######## 修改 self.nucleosome_integrated=[]
        ######## 建立对象 self.final_nucleosome=[]  ########  self.final_nucleosome <- self.nucleosome_integrated
        
        MainProgram.filter_results(self)
        ######## 建立对象 self.final_nucleosome=[]  ########  self.final_nucleosome <- self.nucleosome_integrated

        MainProgram.record_results(self)

        MainProgram.Test_significance( self )
        ######## 建立对象 self.Log_Pvalue_Peak_dict = { }
        ######## 建立对象 self.Log_Pvalue_Valley_dict = { }
        
        ####MainProgram.write_wiggle_file(self)
        ####MainProgram.write_nucleosome_file(self)

        ####def write_wiggle_file(self):
        print('Generating like-wiggle file ......')
        wiggle=open(self.outputfile_wiggle,'w')
        wiggle.write('track type=like_wiggle\nvariableStep chromosome='+self.chromosome+' span=10\n')
        ####wiggle.write('1:coordinate\t2:original_scoring\t3:Gaussian_convolution_smoothing\t4:First_Derivative_of_Gaussian_convolution\t5:LoG\t6:Tirst_Derivative_of_Gaussian_convolution\t7:Nucloesome_determination\n')
        wiggle.write('1:Coordinate\t2:Original_nucleosome_profile\t3:Gaussian_convolution_smoothing\t4:LoG\t5:Minor_LoG\t6:Tag_accumulation\t7:Nucloesomes\n')
        line=0
        for k in range(len(self.score_list)):
            ####wiggle.write(str(k*10+1)+'\t'+str(self.score_list[k])+'\t'+str(self.Gaussian_list[k])+'\t'+str(self.FDoG_list[k])+'\t'+str(self.LoG_list[k])+'\t'+str(self.TDoG_list[k])+'\t'+str(self.secondary_LoG_list[k])+'\t'+str(score_file[k][4])+'\n')
            wiggle.write(str(k*10+1)+'\t'+str(round(self.score_list[k],3))+'\t'+str(round(self.Gaussian_list[k],3))+'\t'+str(round(self.LoG_list[k],3))+'\t'+str(round(self.secondary_LoG_list[k],3))+'\t'+str(self.tag_list[k])+'\t'+str(round(self.score_table[k][4],3))+'\n')
            line=line+1
        wiggle.close()
        print('...... Wiggle file is finished.','\t',time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))
        print('\tChecking ==> Total line of the wiggle file:  ',line)

        ####def write_nucleosome_file(self):
        print('Generating nucleosome collection file ......')
        nucleosome_beginning_ending_file=open(self.outputfile_nucleosome,'w')
        ####nucleosome_beginning_ending_file.write('chromosome='+self.chromosome+'  total_nucleosome:'+str(len(self.final_nucleosome))+'  coordinate:hg18\n')
        nucleosome_beginning_ending_file.write('chromosome='+self.chromosome+'  total_nucleosome:'+str(len(self.final_nucleosome))+'\n')
        ####nucleosome_beginning_ending_file.write('1:Chromosome\t2:Inflection_Pair_Beginning\t3:Inflection_Pair_Ending\t4:Length_between_Inflection\t5:Height_of_nucleosome\t6:Area_under_Curve\t7:Nucleosome_Index\t8:Beginning_Peak_Region\t9:Ending_Peak_Region\t10:Peak_Region_Length\t11:Physical_Property\n')
        nucleosome_beginning_ending_file.write( ('\t').join( ['1:Chromosome','2:Start_inflection','3:End_inflection','4:Nucleosome_index','5:Width_between_inflection','6:Peak_height','7:Area_under_curve','8:Physical_property','9:"-log10(Pvalue_of_peak)"','10:"-log10(Pvalue_of_valley)"'] ) + '\n' )
        line=0
        for k in range(len(self.final_nucleosome)):
            line=line+1
            ### nucleosome=[[pairBC, pairEC, pairBI, pairEI, 'height', 'area', [peakBC,maxBC,maxEC,peakEC,peakBI,maxBI,maxEI,peakEI,BeginningPeakIndex], [peakBC,maxBC,maxEC,peakEC,peakBI,maxBI,maxEI,peakEI,EndingPeakIndex], main/shoulder], ...]
            nucleosome_beginning_ending_file.write(self.chromosome+'\t')    ### 1:Chromosome
            nucleosome_beginning_ending_file.write(str(self.final_nucleosome[k][0])+'\t')    ### 2:Inflection_Pair_Beginning
            nucleosome_beginning_ending_file.write(str(self.final_nucleosome[k][1])+'\t')    ### 3:Inflection_Pair_Ending
            nucleosome_beginning_ending_file.write('Nucleosome:'+str(line)+'\t')    ### 4:Nucleosome_Index
            nucleosome_beginning_ending_file.write(str(self.final_nucleosome[k][1]-self.final_nucleosome[k][0]+10)+'\t')    ### 5:Length_between_Inflection
            nucleosome_beginning_ending_file.write(str(round(self.final_nucleosome[k][4],3))+'\t')    ### 6:Height_of_Nucleosome
            nucleosome_beginning_ending_file.write(str(round(self.final_nucleosome[k][5],3))+'\t')    ### 7:Area_under_Curve
            if self.final_nucleosome[k][8] == 'Main_nucleosome:isolated':
                Physical_Property = 'MainPeak'
            elif self.final_nucleosome[k][8] == 'Main_nucleosome:integrated':
                Physical_Property = 'MainPeak+Shoulder'
            elif self.final_nucleosome[k][8] == 'Main_nucleosome:integrated:doublet':
                Physical_Property = 'MainPeak:doublet'
            elif ( 'shoulder' in self.final_nucleosome[k][8] ) or ( 'Shoulder' in self.final_nucleosome[k][8] ):
                Physical_Property = 'Shoulder'
            else:
                Physical_Property = 'Other'
            nucleosome_beginning_ending_file.write( Physical_Property +'\t')    ### 8:Physical_Property
            nucleosome_beginning_ending_file.write( str( self.Log_Pvalue_Peak_dict[k]   ) + '\t' )    ### 9:"-log10(Pvalue_of_peak)"
            nucleosome_beginning_ending_file.write( str( self.Log_Pvalue_Valley_dict[k] ) + '\n' )    ### 10:"-log10(Pvalue_of_valley)"
        nucleosome_beginning_ending_file.close()
        print('...... Nucleosome collection file is finished.','\t',time.strftime('%Y-%m-%d %A %H:%M:%S', time.localtime()))
        print('\tChecking ==> Total line of the collection file:  ',line)


        print('\n')


def count_chromosome_length( input_file ):
    print('Determine chromosome length ......')
    max_coordinate = 0
    inputTEXT = open( input_file , 'r' )
    while True:
        try:
            line2list = next( inputTEXT ).split('\n')[0].split('\r')[0].split('\t')
            if line2list[ 5 ] == '+':
                right_coordinate = int( line2list[1] ) + 150
            elif line2list[ 5 ] == '-':
                right_coordinate = int( line2list[2] )
            ########
            if right_coordinate > max_coordinate:
                max_coordinate = right_coordinate
            else:
                pass
        except StopIteration:
            break
    inputTEXT.close()
    print('...... Default chromosome length is determined.\n')
    ##return max_coordinate+100
    return max_coordinate


class input_data_pretreatment_for_single_chromosome:
    def __init__( self , parameters_pretreatment_for_single_chromosome ):
        self.chr_recording_dict = { }
        self.chr_recording_list = [ ]
        self.chromosomeName     = parameters_pretreatment_for_single_chromosome[ 'chromosome' ]
        self.chromosome_lines         = 0
        self.chromosome_maxCoordinate = 0
        self.chromosome_lengthSetting = 0
        self.OutputFile_Tags    = parameters_pretreatment_for_single_chromosome[ 'Output_path' ] + self.chromosomeName + '.bed'
        self.OutputFile_summary = parameters_pretreatment_for_single_chromosome[ 'OutputFile_summary' ]
        ########
        if parameters_pretreatment_for_single_chromosome['single_or_paired_setting'] == 's':
            self.do_pretreatment( parameters_pretreatment_for_single_chromosome )
        elif parameters_pretreatment_for_single_chromosome['single_or_paired_setting'] == 'p':
            self.do_pretreatment_for_paired_end( parameters_pretreatment_for_single_chromosome )
        ########
        self.write_down_summary( parameters_pretreatment_for_single_chromosome )

    def do_pretreatment( self , parameters_pretreatment_for_single_chromosome ):
        count_raw_line = 0 #### 计数
        print('Read raw data: ' , parameters_pretreatment_for_single_chromosome[ 'Raw_input_file' ] )
        print('File for preliminary data: ' , self.OutputFile_Tags )
        inputTEXT  = open( parameters_pretreatment_for_single_chromosome[ 'Raw_input_file' ] , 'r' ) #### 只读
        outputTEXT = open( self.OutputFile_Tags , 'w' )
        while True:
            try:
                original_line = next( inputTEXT )
                count_raw_line = count_raw_line + 1  #### 计数
                # sys.stdout.write('\r    Raw line: '+str(count_raw_line) )
                line2list = original_line.split('\n')[0].split('\r')[0].split('\t')
                ####
                if line2list[ 5 ] == '+':
                    right_coordinate = int( line2list[1] ) + 150
                elif line2list[ 5 ] == '-':
                    right_coordinate = int( line2list[2] )
                ####
                chromosome = line2list[ 0 ]
                self.chr_recording_dict[ chromosome ] = ''
                ##
                if self.chromosomeName == chromosome:
                    outputTEXT.write( original_line )
                    self.chromosome_lines = self.chromosome_lines + 1
                    if right_coordinate > self.chromosome_maxCoordinate:
                        self.chromosome_maxCoordinate = right_coordinate
                    else:
                        pass
                else:
                    pass
                ####
                # sys.stdout.flush()
            except StopIteration:
                break
        inputTEXT.close() #### 读毕
        outputTEXT.close()
        print('\tChecking ==> Lines of raw data: ' , count_raw_line )
        print('\tChecking ==> Lines for ' , self.chromosomeName , ': ' , self.chromosome_lines )
        if parameters_pretreatment_for_single_chromosome[ 'length' ] == 'none':
            self.chromosome_lengthSetting = self.chromosome_maxCoordinate
        elif parameters_pretreatment_for_single_chromosome[ 'length' ] != 'none':
            self.chromosome_lengthSetting = parameters_pretreatment_for_single_chromosome[ 'length' ]
        ####
        self.chr_recording_list = list( self.chr_recording_dict.keys() )

    def do_pretreatment_for_paired_end( self , parameters_pretreatment_for_single_chromosome ):
        count_raw_line = 0 #### 计数
        print('Read raw data: ' , parameters_pretreatment_for_single_chromosome[ 'Raw_input_file' ] )
        print('File for preliminary data: ' , self.OutputFile_Tags )
        inputTEXT  = open( parameters_pretreatment_for_single_chromosome[ 'Raw_input_file' ] , 'r' ) #### 只读
        outputTEXT = open( self.OutputFile_Tags , 'w' )
        while True:
            try:
                original_line = next( inputTEXT )
                count_raw_line = count_raw_line + 1  #### 计数
                # sys.stdout.write('\r    Raw line: '+str(count_raw_line) )
                line2list = original_line.split('\n')[0].split('\r')[0].split('\t')
                ####
                ##if line2list[ 5 ] == '+':
                ##    right_coordinate = int( line2list[1] ) + 150
                ##elif line2list[ 5 ] == '-':
                ##    right_coordinate = int( line2list[2] )
                right_coordinate = int( line2list[2] )
                ####
                chromosome = line2list[ 0 ]
                self.chr_recording_dict[ chromosome ] = ''
                ##
                if self.chromosomeName == chromosome:
                    outputTEXT.write( original_line )
                    self.chromosome_lines = self.chromosome_lines + 1
                    if right_coordinate > self.chromosome_maxCoordinate:
                        self.chromosome_maxCoordinate = right_coordinate
                    else:
                        pass
                else:
                    pass
                ####
                # sys.stdout.flush()
            except StopIteration:
                break
        inputTEXT.close() #### 读毕
        outputTEXT.close()
        print('\tChecking ==> Lines of raw data: ' , count_raw_line )
        print('\tChecking ==> Lines for ' , self.chromosomeName , ': ' , self.chromosome_lines )
        if parameters_pretreatment_for_single_chromosome[ 'length' ] == 'none':
            self.chromosome_lengthSetting = self.chromosome_maxCoordinate
        elif parameters_pretreatment_for_single_chromosome[ 'length' ] != 'none':
            self.chromosome_lengthSetting = parameters_pretreatment_for_single_chromosome[ 'length' ]
        ####
        self.chr_recording_list = list( self.chr_recording_dict.keys() )

    def write_down_summary( self , parameters_pretreatment_for_single_chromosome ):
        print('Write down summary: ' , self.OutputFile_summary )
        outputTEXT = open( self.OutputFile_summary , 'w' )
        outputTEXT.write( ('\t').join( [ 'Chromosome' , 'Tags' , 'Max_coordinate' , 'Recorded_chromosome_length' ] ) + '\n' )
        outputTEXT.write( ('\t').join( [ self.chromosomeName , str(self.chromosome_lines ) , str(self.chromosome_maxCoordinate) , str(self.chromosome_lengthSetting) ] ) + '\n' )
        outputTEXT.close()


class input_data_pretreatment:
    def __init__( self , parameters_pretreatment ):
        self.chr_list = [ ]
        self.chr2maxCoordinate_dict = { }
        self.chr2lengthSetting_dict = { }
        self.chr2lines_dict = { }
        self.chr2outputfile_dict = { }
        ########
        if parameters_pretreatment['single_or_paired_setting'] == 's':
            self.do_pretreatment( parameters_pretreatment )
        elif parameters_pretreatment['single_or_paired_setting'] == 'p':
            self.do_pretreatment_for_paired_end( parameters_pretreatment )
        ########
        self.write_down_summary( parameters_pretreatment )

    def do_pretreatment(self , parameters_pretreatment):
        chr2outputTEXT_dict = { }
        count_raw_line = 0 #### 计数
        print('Read raw data: ' , parameters_pretreatment['Raw_input_file'] )
        print('Path to record preliminary data: ' , parameters_pretreatment['Output_path'] )
        inputTEXT = open( parameters_pretreatment['Raw_input_file'] , 'r' ) #### 只读
        while True:
            try:
                original_line = next( inputTEXT )
                count_raw_line = count_raw_line + 1  #### 计数
                # sys.stdout.write('\r    Raw line: '+str(count_raw_line) )
                line2list = original_line.split('\n')[0].split('\r')[0].split('\t')
                ####
                if line2list[ 5 ] == '+':
                    right_coordinate = int( line2list[1] ) + 150
                elif line2list[ 5 ] == '-':
                    right_coordinate = int( line2list[2] )
                ####
                chromosome = line2list[ 0 ]
                if chromosome not in self.chr2outputfile_dict.keys():
                    self.chr2outputfile_dict[ chromosome ] = parameters_pretreatment['Output_path']+chromosome+'.bed'
                    chr2outputTEXT_dict[ chromosome ] = open( self.chr2outputfile_dict[ chromosome ] , 'w' )
                    self.chr2maxCoordinate_dict[ chromosome ] = 0
                    self.chr2lines_dict[ chromosome ] = 0
                    self.chr_list.append( chromosome )
                elif chromosome in self.chr2outputfile_dict.keys():
                    pass
                ################
                chr2outputTEXT_dict[ chromosome ].write( original_line )
                self.chr2lines_dict[ chromosome ] = self.chr2lines_dict[ chromosome ] + 1
                ####
                if right_coordinate > self.chr2maxCoordinate_dict[ chromosome ]:
                    self.chr2maxCoordinate_dict[ chromosome ] = right_coordinate
                else:
                    pass
                ####
                # sys.stdout.flush()
            except StopIteration:
                break
        inputTEXT.close() #### 读毕
        print('\tChecking ==> Lines of raw data: ' , count_raw_line )
        ########
        for chromosome in self.chr_list:
            chr2outputTEXT_dict[ chromosome ].close()
            ##self.chr2lengthSetting_dict[ chromosome ] = self.chr2maxCoordinate_dict[ chromosome ] + 100
            self.chr2lengthSetting_dict[ chromosome ] = self.chr2maxCoordinate_dict[ chromosome ]
        ########

    def do_pretreatment_for_paired_end( self , parameters_pretreatment ):
        chr2outputTEXT_dict = { }
        count_raw_line = 0 #### 计数
        print('Read raw data: ' , parameters_pretreatment['Raw_input_file'] )
        print('Path to record preliminary data: ' , parameters_pretreatment['Output_path'] )
        inputTEXT = open( parameters_pretreatment['Raw_input_file'] , 'r' ) #### 只读
        while True:
            try:
                original_line = next( inputTEXT )
                count_raw_line = count_raw_line + 1  #### 计数
                # sys.stdout.write('\r    Raw line: '+str(count_raw_line) )
                line2list = original_line.split('\n')[0].split('\r')[0].split('\t')
                ####
                ##if line2list[ 5 ] == '+':
                ##    right_coordinate = int( line2list[1] ) + 150
                ##elif line2list[ 5 ] == '-':
                ##    right_coordinate = int( line2list[2] )
                right_coordinate = int( line2list[2] )
                ####
                chromosome = line2list[ 0 ]
                if chromosome not in self.chr2outputfile_dict.keys():
                    self.chr2outputfile_dict[ chromosome ] = parameters_pretreatment['Output_path']+chromosome+'.bed'
                    chr2outputTEXT_dict[ chromosome ] = open( self.chr2outputfile_dict[ chromosome ] , 'w' )
                    self.chr2maxCoordinate_dict[ chromosome ] = 0
                    self.chr2lines_dict[ chromosome ] = 0
                    self.chr_list.append( chromosome )
                elif chromosome in self.chr2outputfile_dict.keys():
                    pass
                ################
                chr2outputTEXT_dict[ chromosome ].write( original_line )
                self.chr2lines_dict[ chromosome ] = self.chr2lines_dict[ chromosome ] + 1
                ####
                if right_coordinate > self.chr2maxCoordinate_dict[ chromosome ]:
                    self.chr2maxCoordinate_dict[ chromosome ] = right_coordinate
                else:
                    pass
                ####
                # sys.stdout.flush()
            except StopIteration:
                break
        inputTEXT.close() #### 读毕
        print('\tChecking ==> Lines of raw data: ' , count_raw_line )
        ########
        for chromosome in self.chr_list:
            chr2outputTEXT_dict[ chromosome ].close()
            ##self.chr2lengthSetting_dict[ chromosome ] = self.chr2maxCoordinate_dict[ chromosome ] + 100
            self.chr2lengthSetting_dict[ chromosome ] = self.chr2maxCoordinate_dict[ chromosome ]
        ########

    def write_down_summary( self , parameters_pretreatment ):
        print('Write down summary: ' , parameters_pretreatment['OutputFile_summary'] )
        outputTEXT = open( parameters_pretreatment['OutputFile_summary'] , 'w' )
        outputTEXT.write( ('\t').join( [ 'Chromosome' , 'Tags' , 'Max_coordinate' , 'Recorded_chromosome_length' ] ) + '\n' )
        for chromosome in sorted( self.chr_list ):
            outputTEXT.write( ('\t').join( [ chromosome , str(self.chr2lines_dict[ chromosome ]) , str(self.chr2maxCoordinate_dict[ chromosome ]) , str(self.chr2lengthSetting_dict[ chromosome ]) ] ) + '\n' )
        outputTEXT.close()


def Merge_overall_results( parameters_Merge_results ):
    print('Gather all nucleosome detection results: ' , parameters_Merge_results['OutputFile'] )
    outputTEXT = open( parameters_Merge_results['OutputFile'] , 'w' )
    total_nucleosome____dict = { }
    headline2 = ''
    for chromosome in sorted( parameters_Merge_results['Chromosome_list'] ):
        InputFile = parameters_Merge_results['Input_Files'] + '_' + chromosome + '.like_bed'
        inputTEXT = open( InputFile , 'r' ) #### 只读
        headline1 = next( inputTEXT ) #### headline1
        outputTEXT.write( headline1 ) #### 记录
        headline2 = next( inputTEXT ) #### headline2
        inputTEXT.close() #### 读毕
    outputTEXT.write( headline2 )     #### 记录
    ########
    for chromosome in sorted( parameters_Merge_results['Chromosome_list'] ):
        InputFile = parameters_Merge_results['Input_Files'] + '_' + chromosome + '.like_bed'
        print('Read from: ' , InputFile )
        inputTEXT = open( InputFile , 'r' ) #### 只读
        next( inputTEXT ) #### headline1
        next( inputTEXT ) #### headline2
        while True:
            try:
                original_line = next( inputTEXT )
                outputTEXT.write( original_line )     #### 记录
            except StopIteration:
                break
        inputTEXT.close() #### 读毕
    outputTEXT.close()
    print('...... All the nucleosome results are gathered.')
    print('\n')



if __name__=='__main__':
    
    print('\n\nProgram to detect nucleosomes from MNase-seq data.')
    print('\nThe program should run in Python3.\n')
    usage='usage: iNPS.py [options]'
    parser = OptionParser( usage = '%prog  -i /path/inputfile  -o /path/outputfile  -c chromosome_name  -l chromosome_length  --s_p single_or_paired_end_data' , version = '%prog Version:1.2.2' )
    parser.add_option('-i', '--input',      dest='input_file',                        type='string', help='"/path/filename"  INPUT_FILE file of single-end sequencing tags in a standard BED format ( chromosome <tab> start <tab> end <tab> name <tab> score <tab> strand ), or paired-end tags in a 3-column BED format ( chromosome <tab> start <tab> end ).')
    parser.add_option('-o', '--output',     dest='output_file',                       type='string', help='"/path/filename"  Here, the name extension is unnecessary. Software will output two result files, "filename_[ChromosomeName].like_bed" and "filename_[ChromosomeName].like_wig", to record coordinates and profiles of detected nucleosomes respectively. The chromosome name will be added as suffix in the file names. If your detect nucleosomes on multiple chromosomes, for each chromosome, software will output two result files "filename_[ChromosomeName].like_bed" and "filename_[ChromosomeName].like_wig" respectively. And finally, a file "filename_Gathering.like_bed" will gather the detected nucleosomes on every chromosome. Note that a path "/path/filename/" or "/path/filename_[ChromosomeName]/" will be built to record the preliminary and intermediate data.')
    parser.add_option('-c', '--chrname',    dest='chromosome_name',                   type='string', help='Specify the name (or abbreviation) of the chromosome, if you would like to do nucleosome detection ONLY on ONE single chromosome. For nucleosome detection on multiple chromosomes, please do NOT use this parameter. That is, if your do NOT use this parameter, software will detect nucleosome on each chromosome ONE-BY-ONE in the input data as default.')
    parser.add_option('-l', '--chrlength',  dest='chromosome_length',                 type='int',    help='The length of the chromosome specified by parameter "-c" or "--chrname". ONLY used for nucleosome detection on ONE single chromosome (parameter "-c" or "--chrname" is setted). If you do NOT use this parameter, software will find the maximum coordinate in the input data to represent the chromosome length as default. For nucleosome detection on multiple chromosomes, please do NOT use this parameter. The length of each chromosome will be determined by the tag with maximum coordinate of the corresponding chromosome respectively.')
    parser.add_option('--s_p',              dest='single_or_paired_end',              type='string', default='s' , help='"s" or "p". [Default = s] Set to "p" if the input data is paired-end tags. Otherwise, set to "s" or use the default setting if the input data is single-end tags.')
    parser.add_option('--pe_max',           dest='superior_limit_of_paired_end_tags', type='int',    default=200 , help='The superior limit of the length of paired-end tags. [Default = 200] The tags longer than the cutoff will be ignored. This parameter is ONLY available for paired-end sequencing data. Please avoid using too large value.')
    parser.add_option('--pe_min',           dest='inferior_limit_of_paired_end_tags', type='int',    default=100 , help='The inferior limit of the length of paired-end tags. [Default = 100] The tags shorter than the cutoff will be ignored. This parameter is ONLY available for paired-end sequencing data. Please avoid using too small value.')
    (options, args) = parser.parse_args( )
    ########
    if options.input_file == None:
        parser.error('-h for help or provide the input file name!')
    else:
        pass
    ########
    if options.output_file == None:
        parser.error('-h for help or provide the output file name!')
    else:
        pass
    ########
    if options.single_or_paired_end == 's':
        FilesParameters[ 'single_or_paired' ] = 's'
    elif options.single_or_paired_end == 'p':
        FilesParameters[ 'single_or_paired' ] = 'p'
        FilesParameters[ 'pe_max' ] = options.superior_limit_of_paired_end_tags
        FilesParameters[ 'pe_min' ] = options.inferior_limit_of_paired_end_tags
    else:
        parser.error('-h for help or correctly specify the type of input data: single-end ("s") or paired_end ("p")')





    ########
    print('Prepare Gaussian function and Laplacian of Gaussian operator for convolution ......')
    ConvolutionParameters=Gaussian_profile(PreliminaryConvolutionParameters)
    ####print('\tDiscrete Gaussian function:\t(Range:',len(ConvolutionParameters['Gaussian']),')')
    ####print('\t\t','Gaussian','\t\t','1st_Der','\t\t','2nd_Der','\t\t','3rd_Der')
    ####for i in range(len(ConvolutionParameters['Gaussian'])):
    ####    print('\t\t',round(ConvolutionParameters['Gaussian'][i],6),'\t\t',round(ConvolutionParameters['First_Derivative_of_Gaussian'][i],6),'\t\t',round(ConvolutionParameters['LoG'][i],6),'\t\t',round(ConvolutionParameters['Third_Derivative_of_Gaussian'][i],6))
    ####print('\tDiscrete secondary Gaussian function:\t(Range:',len(ConvolutionParameters['secondary_Gaussian']),')')
    ####print('\t\t','Gaussian','\t\t','1st_Der','\t\t','2nd_Der','\t\t','3rd_Der')
    ####for i in range(len(ConvolutionParameters['secondary_Gaussian'])):
    ####    print('\t\t',round(ConvolutionParameters['secondary_Gaussian'][i],6),'\t\t',round(ConvolutionParameters['secondary_First_Derivative_of_Gaussian'][i],6),'\t\t',round(ConvolutionParameters['secondary_LoG'][i],6),'\t\t',round(ConvolutionParameters['secondary_Third_Derivative_of_Gaussian'][i],6))
    print('...... The preparation of Gaussian convolution parameters is finished.') 
    print('\n')
    ########





    if options.chromosome_name != None:
        print('Do nucleosome detection on chromosome: ' , options.chromosome_name )
        print('\n')
        ######## inter-mediate path ########
        Intermediate_Path = options.output_file+'_'+options.chromosome_name+'/' #### inter-mediate
        if os.path.exists( Intermediate_Path ):
            pass
        else:
            os.makedirs( Intermediate_Path )
        ######## Data pretreatment ########
        print('Do data pretreatment ......')
        parameters_pretreatment_for_single_chromosome = { 'Raw_input_file': options.input_file,
                                                          'Output_path':    Intermediate_Path,
                                                          'OutputFile_summary': Intermediate_Path+'InputData_Summary.txt',
                                                          'chromosome':   options.chromosome_name ,
                                                          'single_or_paired_setting': options.single_or_paired_end,
                                                          }
        if options.chromosome_length == None:
            parameters_pretreatment_for_single_chromosome[ 'length' ] = 'none'
        elif options.chromosome_length != None:
            parameters_pretreatment_for_single_chromosome[ 'length' ] = options.chromosome_length
        InputData_Summary_for_single_chromosome = input_data_pretreatment_for_single_chromosome( parameters_pretreatment_for_single_chromosome ) #### get the summary of input data
        ####    InputData_Summary_for_single_chromosome.chr_recording_dict
        ####    InputData_Summary_for_single_chromosome.chr_recording_list
        ####    InputData_Summary_for_single_chromosome.chromosomeName
        ####    InputData_Summary_for_single_chromosome.chromosome_lines
        ####    InputData_Summary_for_single_chromosome.chromosome_maxCoordinate
        ####    InputData_Summary_for_single_chromosome.chromosome_lengthSetting
        ####    InputData_Summary_for_single_chromosome.OutputFile_Tags
        ####    InputData_Summary_for_single_chromosome.OutputFile_summary
        print('...... Data pretreatment finished.')
        print('\n')
        ########
        if ( options.chromosome_name in InputData_Summary_for_single_chromosome.chr_recording_list ) and ( InputData_Summary_for_single_chromosome.chromosome_lines > 0 ):
            FilesParameters[ 'Chosen_Chromosome_Abbreviation' ] = options.chromosome_name
            FilesParameters[ 'inputfile_for_nucleosome_positioning' ] = InputData_Summary_for_single_chromosome.OutputFile_Tags
            FilesParameters[ 'outputfile_like_bed' ] = options.output_file + '_' + options.chromosome_name + '.like_bed'
            FilesParameters[ 'outputfile_like_wig' ] = options.output_file + '_' + options.chromosome_name + '.like_wig'
            if options.chromosome_length == None:
                FilesParameters[ 'Chosen_Chromosome_Length' ] = InputData_Summary_for_single_chromosome.chromosome_lengthSetting
            elif options.chromosome_length != None:
                FilesParameters[ 'Chosen_Chromosome_Length' ] = options.chromosome_length
            #### 开始执行函数 ####
            #### 开始执行函数 ####
            #### 开始执行函数 ####
            NucleosomeAccuratePositioning( FilesParameters , ConvolutionParameters , threshold )
            #### 执行函数完毕 ####
            #### 执行函数完毕 ####
            #### 执行函数完毕 ####
        else:
            print('Please provide correct chromosome name, or choose chromosome among: ' , (',').join(sorted(InputData_Summary_for_single_chromosome.chr_recording_list)) )
    ################
    ################
    elif options.chromosome_name == None:
        ######## inter-mediate path ########
        Intermediate_Path = options.output_file+'/' #### inter-mediate
        if os.path.exists( Intermediate_Path ):
            pass
        else:
            os.makedirs( Intermediate_Path )
        ######## Data pretreatment ########
        print('Do data pretreatment ......')
        parameters_pretreatment = { 'Raw_input_file': options.input_file,
                                    'Output_path':    Intermediate_Path,
                                    'OutputFile_summary': Intermediate_Path+'InputData_Summary.txt',
                                    'single_or_paired_setting': options.single_or_paired_end,
                                    }
        InputData_Summary = input_data_pretreatment( parameters_pretreatment ) #### get the summary of input data
        ####    InputData_Summary.chr_list = [ ]
        ####    InputData_Summary.chr2maxCoordinate_dict = { }
        ####    InputData_Summary.chr2lengthSetting_dict = { }
        ####    InputData_Summary.chr2lines_dict = { }
        ####    InputData_Summary.chr2outputfile_dict = { }
        print('...... Data pretreatment finished.')
        ########
        for each_chromosome in sorted(InputData_Summary.chr_list):
            FilesParameters[ 'Chosen_Chromosome_Abbreviation' ] = each_chromosome
            FilesParameters[ 'inputfile_for_nucleosome_positioning' ] = InputData_Summary.chr2outputfile_dict[ each_chromosome ]
            FilesParameters[ 'outputfile_like_bed' ] = options.output_file + '_' + each_chromosome + '.like_bed'
            FilesParameters[ 'outputfile_like_wig' ] = options.output_file + '_' + each_chromosome + '.like_wig'
            FilesParameters[ 'Chosen_Chromosome_Length' ] = InputData_Summary.chr2lengthSetting_dict[ each_chromosome ]
            #### 开始执行函数 ####
            #### 开始执行函数 ####
            #### 开始执行函数 ####
            NucleosomeAccuratePositioning( FilesParameters , ConvolutionParameters , threshold )
            #### 执行函数完毕 ####
            #### 执行函数完毕 ####
            #### 执行函数完毕 ####
        ######## 合并 nucleosomes ########
        parameters_Merge_results = { 'Input_Files': options.output_file,
                                     'Chromosome_list': InputData_Summary.chr_list,
                                     'OutputFile':  options.output_file + '_Gathering.like_bed',
                                     }
        print('\n')
        Merge_overall_results( parameters_Merge_results )
    ########





print('\nThe end.\n\n')



