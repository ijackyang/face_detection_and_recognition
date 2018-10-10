import numpy as np
import os
import pickle
import cv2

def clip_min(x1):
        return max(x1,0) 

def clip_max(x1):
        return min(x1,250)

def bbox_mirror(bbox):
        return np.array([249-bbox[2],bbox[1],249-bbox[0],bbox[3]])

def generate_recognition_dataset(dataset='train',low=0,high=0,data=None,face_ids=None):
        ff = open('bbox_ids_'+dataset+'.txt','w+')

        for id in range(low,high):
        	img_1  = cv2.imread(data[face_ids[16*id+0]][1])
        	img_2  = cv2.imread(data[face_ids[16*id+1]][1])
        	img_3  = cv2.imread(data[face_ids[16*id+2]][1])
        	img_4  = cv2.imread(data[face_ids[16*id+3]][1])
        	img_5  = cv2.imread(data[face_ids[16*id+4]][1])
        	img_6  = cv2.imread(data[face_ids[16*id+5]][1])
        	img_7  = cv2.imread(data[face_ids[16*id+6]][1])
        	img_8  = cv2.imread(data[face_ids[16*id+7]][1])
        	img_9  = cv2.imread(data[face_ids[16*id+8]][1])
        	img_10 = cv2.imread(data[face_ids[16*id+9]][1])
        	img_11 = cv2.imread(data[face_ids[16*id+10]][1])
        	img_12 = cv2.imread(data[face_ids[16*id+11]][1])
        	img_13 = cv2.imread(data[face_ids[16*id+12]][1])
        	img_14 = cv2.imread(data[face_ids[16*id+13]][1])
        	img_15 = cv2.imread(data[face_ids[16*id+14]][1])
        	img_16 = cv2.imread(data[face_ids[16*id+15]][1])
        
        	new_img = np.zeros((1024,1024,3),dtype=np.uint8)
        	new_img[0:250,   0:250,:]    	= img_1
        	new_img[256:506, 0:250,:]  	= img_2
        	new_img[512:762, 0:250,:]  	= img_3
        	new_img[768:1018,0:250,:] 	= img_4
        	new_img[0:250,   256:506,:]     = img_5
        	new_img[256:506, 256:506,:]  	= img_6
        	new_img[512:762, 256:506,:]  	= img_7
        	new_img[768:1018,256:506,:] 	= img_8
        	new_img[0:250,   512:762,:]    	= img_9
        	new_img[256:506, 512:762:]  	= img_10
        	new_img[512:762, 512:762:]  	= img_11
        	new_img[768:1018,512:762:] 	= img_12
        	new_img[0:250,   768:1018,:]    = img_13
        	new_img[256:506, 768:1018,:]  	= img_14
        	new_img[512:762, 768:1018,:]  	= img_15
        	new_img[768:1018,768:1018,:] 	= img_16
        
                ff.write('0/'+str(id)+'.jpg'+'\n'+'16\n')
                id_1  = data[face_ids[16*id+0 ]][0]
                bb_1  = data[face_ids[16*id+0 ]][2]
                id_2  = data[face_ids[16*id+1 ]][0]
                bb_2  = data[face_ids[16*id+1 ]][2]
                id_3  = data[face_ids[16*id+2 ]][0]
                bb_3  = data[face_ids[16*id+2 ]][2]
                id_4  = data[face_ids[16*id+3 ]][0]
                bb_4  = data[face_ids[16*id+3 ]][2]
                id_5  = data[face_ids[16*id+4 ]][0]
                bb_5  = data[face_ids[16*id+4 ]][2]
                id_6  = data[face_ids[16*id+5 ]][0]
                bb_6  = data[face_ids[16*id+5 ]][2]
                id_7  = data[face_ids[16*id+6 ]][0]
                bb_7  = data[face_ids[16*id+6 ]][2]
                id_8  = data[face_ids[16*id+7 ]][0]
                bb_8  = data[face_ids[16*id+7 ]][2]
                id_9  = data[face_ids[16*id+8 ]][0]
                bb_9  = data[face_ids[16*id+8 ]][2]
                id_10 = data[face_ids[16*id+9 ]][0]
                bb_10 = data[face_ids[16*id+9 ]][2]
                id_11 = data[face_ids[16*id+10]][0]
                bb_11 = data[face_ids[16*id+10]][2]
                id_12 = data[face_ids[16*id+11]][0]
                bb_12 = data[face_ids[16*id+11]][2]
                id_13 = data[face_ids[16*id+12]][0]
                bb_13 = data[face_ids[16*id+12]][2]
                id_14 = data[face_ids[16*id+13]][0]
                bb_14 = data[face_ids[16*id+13]][2]
                id_15 = data[face_ids[16*id+14]][0]
                bb_15 = data[face_ids[16*id+14]][2]
                id_16 = data[face_ids[16*id+15]][0]
                bb_16 = data[face_ids[16*id+15]][2]
         
                bb_1   = [clip_min(bb_1[0])+256*0,clip_min(bb_1[1])+256*0,clip_max(bb_1[2])-clip_min(bb_1[0]),clip_max(bb_1[3])-clip_min(bb_1[1])]
                bb_2   = [clip_min(bb_2[0])+256*0,clip_min(bb_2[1])+256*1,clip_max(bb_2[2])-clip_min(bb_2[0]),clip_max(bb_2[3])-clip_min(bb_2[1])]
                bb_3   = [clip_min(bb_3[0])+256*0,clip_min(bb_3[1])+256*2,clip_max(bb_3[2])-clip_min(bb_3[0]),clip_max(bb_3[3])-clip_min(bb_3[1])]
                bb_4   = [clip_min(bb_4[0])+256*0,clip_min(bb_4[1])+256*3,clip_max(bb_4[2])-clip_min(bb_4[0]),clip_max(bb_4[3])-clip_min(bb_4[1])]
                bb_5   = [clip_min(bb_5[0])+256*1,clip_min(bb_5[1])+256*0,clip_max(bb_5[2])-clip_min(bb_5[0]),clip_max(bb_5[3])-clip_min(bb_5[1])]
                bb_6   = [clip_min(bb_6[0])+256*1,clip_min(bb_6[1])+256*1,clip_max(bb_6[2])-clip_min(bb_6[0]),clip_max(bb_6[3])-clip_min(bb_6[1])]
                bb_7   = [clip_min(bb_7[0])+256*1,clip_min(bb_7[1])+256*2,clip_max(bb_7[2])-clip_min(bb_7[0]),clip_max(bb_7[3])-clip_min(bb_7[1])]
                bb_8   = [clip_min(bb_8[0])+256*1,clip_min(bb_8[1])+256*3,clip_max(bb_8[2])-clip_min(bb_8[0]),clip_max(bb_8[3])-clip_min(bb_8[1])]
                bb_9   = [clip_min(bb_9[0])+256*2,clip_min(bb_9[1])+256*0,clip_max(bb_9[2])-clip_min(bb_9[0]),clip_max(bb_9[3])-clip_min(bb_9[1])]
                bb_10  = [clip_min(bb_10[0])+256*2,clip_min(bb_10[1])+256*1,clip_max(bb_10[2])-clip_min(bb_10[0]),clip_max(bb_10[3])-clip_min(bb_10[1])]
                bb_11  = [clip_min(bb_11[0])+256*2,clip_min(bb_11[1])+256*2,clip_max(bb_11[2])-clip_min(bb_11[0]),clip_max(bb_11[3])-clip_min(bb_11[1])]
                bb_12  = [clip_min(bb_12[0])+256*2,clip_min(bb_12[1])+256*3,clip_max(bb_12[2])-clip_min(bb_12[0]),clip_max(bb_12[3])-clip_min(bb_12[1])]
                bb_13  = [clip_min(bb_13[0])+256*3,clip_min(bb_13[1])+256*0,clip_max(bb_13[2])-clip_min(bb_13[0]),clip_max(bb_13[3])-clip_min(bb_13[1])]
                bb_14  = [clip_min(bb_14[0])+256*3,clip_min(bb_14[1])+256*1,clip_max(bb_14[2])-clip_min(bb_14[0]),clip_max(bb_14[3])-clip_min(bb_14[1])]
                bb_15  = [clip_min(bb_15[0])+256*3,clip_min(bb_15[1])+256*2,clip_max(bb_15[2])-clip_min(bb_15[0]),clip_max(bb_15[3])-clip_min(bb_15[1])]
                bb_16  = [clip_min(bb_16[0])+256*3,clip_min(bb_16[1])+256*3,clip_max(bb_16[2])-clip_min(bb_16[0]),clip_max(bb_16[3])-clip_min(bb_16[1])]
         
                ff.write(' '.join(str(x) for x in bb_1 )+' '+str(id_1 )+'\n')
                ff.write(' '.join(str(x) for x in bb_2 )+' '+str(id_2 )+'\n')
                ff.write(' '.join(str(x) for x in bb_3 )+' '+str(id_3 )+'\n')
                ff.write(' '.join(str(x) for x in bb_4 )+' '+str(id_4 )+'\n')
                ff.write(' '.join(str(x) for x in bb_5 )+' '+str(id_5 )+'\n')
                ff.write(' '.join(str(x) for x in bb_6 )+' '+str(id_6 )+'\n')
                ff.write(' '.join(str(x) for x in bb_7 )+' '+str(id_7 )+'\n')
                ff.write(' '.join(str(x) for x in bb_8 )+' '+str(id_8 )+'\n')
                ff.write(' '.join(str(x) for x in bb_9 )+' '+str(id_9 )+'\n')
                ff.write(' '.join(str(x) for x in bb_10)+' '+str(id_10)+'\n')
                ff.write(' '.join(str(x) for x in bb_11)+' '+str(id_11)+'\n')
                ff.write(' '.join(str(x) for x in bb_12)+' '+str(id_12)+'\n')
                ff.write(' '.join(str(x) for x in bb_13)+' '+str(id_13)+'\n')
                ff.write(' '.join(str(x) for x in bb_14)+' '+str(id_14)+'\n')
                ff.write(' '.join(str(x) for x in bb_15)+' '+str(id_15)+'\n')
                ff.write(' '.join(str(x) for x in bb_16)+' '+str(id_16)+'\n')
        
        	cv2.imwrite('../../recognition_dataset_new/'+dataset+'/0/'+str(id)+'.jpg',new_img)


        	new_img = np.zeros((1024,1024,3),dtype=np.uint8)
        	new_img[0:250,   0:250,:]    	= cv2.flip(img_1,1)
        	new_img[256:506, 0:250,:]  	= cv2.flip(img_2,1)
        	new_img[512:762, 0:250,:]  	= cv2.flip(img_3,1)
        	new_img[768:1018,0:250,:] 	= cv2.flip(img_4,1)
        	new_img[0:250,   256:506,:]     = cv2.flip(img_5,1)
        	new_img[256:506, 256:506,:]  	= cv2.flip(img_6,1)
        	new_img[512:762, 256:506,:]  	= cv2.flip(img_7,1)
        	new_img[768:1018,256:506,:] 	= cv2.flip(img_8,1)
        	new_img[0:250,   512:762,:]    	= cv2.flip(img_9,1)
        	new_img[256:506, 512:762:]  	= cv2.flip(img_10,1)
        	new_img[512:762, 512:762:]  	= cv2.flip(img_11,1)
        	new_img[768:1018,512:762:] 	= cv2.flip(img_12,1)
        	new_img[0:250,   768:1018,:]    = cv2.flip(img_13,1)
        	new_img[256:506, 768:1018,:]  	= cv2.flip(img_14,1)
        	new_img[512:762, 768:1018,:]  	= cv2.flip(img_15,1)
        	new_img[768:1018,768:1018,:] 	= cv2.flip(img_16,1)
        
                ff.write('1/'+str(id)+'.jpg'+'\n'+'16\n')
                id_1  = data[face_ids[16*id+0 ]][0]
                bb_1  = bbox_mirror(data[face_ids[16*id+0 ]][2])
                id_2  = data[face_ids[16*id+1 ]][0]
                bb_2  = bbox_mirror(data[face_ids[16*id+1 ]][2])
                id_3  = data[face_ids[16*id+2 ]][0]
                bb_3  = bbox_mirror(data[face_ids[16*id+2 ]][2])
                id_4  = data[face_ids[16*id+3 ]][0]
                bb_4  = bbox_mirror(data[face_ids[16*id+3 ]][2])
                id_5  = data[face_ids[16*id+4 ]][0]
                bb_5  = bbox_mirror(data[face_ids[16*id+4 ]][2])
                id_6  = data[face_ids[16*id+5 ]][0]
                bb_6  = bbox_mirror(data[face_ids[16*id+5 ]][2])
                id_7  = data[face_ids[16*id+6 ]][0]
                bb_7  = bbox_mirror(data[face_ids[16*id+6 ]][2])
                id_8  = data[face_ids[16*id+7 ]][0]
                bb_8  = bbox_mirror(data[face_ids[16*id+7 ]][2])
                id_9  = data[face_ids[16*id+8 ]][0]
                bb_9  = bbox_mirror(data[face_ids[16*id+8 ]][2])
                id_10 = data[face_ids[16*id+9 ]][0]
                bb_10 = bbox_mirror(data[face_ids[16*id+9 ]][2])
                id_11 = data[face_ids[16*id+10]][0]
                bb_11 = bbox_mirror(data[face_ids[16*id+10]][2])
                id_12 = data[face_ids[16*id+11]][0]
                bb_12 = bbox_mirror(data[face_ids[16*id+11]][2])
                id_13 = data[face_ids[16*id+12]][0]
                bb_13 = bbox_mirror(data[face_ids[16*id+12]][2])
                id_14 = data[face_ids[16*id+13]][0]
                bb_14 = bbox_mirror(data[face_ids[16*id+13]][2])
                id_15 = data[face_ids[16*id+14]][0]
                bb_15 = bbox_mirror(data[face_ids[16*id+14]][2])
                id_16 = data[face_ids[16*id+15]][0]
                bb_16 = bbox_mirror(data[face_ids[16*id+15]][2])
         
                bb_1   = [clip_min(bb_1[0])+256*0,clip_min(bb_1[1])+256*0,clip_max(bb_1[2])-clip_min(bb_1[0]),clip_max(bb_1[3])-clip_min(bb_1[1])]
                bb_2   = [clip_min(bb_2[0])+256*0,clip_min(bb_2[1])+256*1,clip_max(bb_2[2])-clip_min(bb_2[0]),clip_max(bb_2[3])-clip_min(bb_2[1])]
                bb_3   = [clip_min(bb_3[0])+256*0,clip_min(bb_3[1])+256*2,clip_max(bb_3[2])-clip_min(bb_3[0]),clip_max(bb_3[3])-clip_min(bb_3[1])]
                bb_4   = [clip_min(bb_4[0])+256*0,clip_min(bb_4[1])+256*3,clip_max(bb_4[2])-clip_min(bb_4[0]),clip_max(bb_4[3])-clip_min(bb_4[1])]
                bb_5   = [clip_min(bb_5[0])+256*1,clip_min(bb_5[1])+256*0,clip_max(bb_5[2])-clip_min(bb_5[0]),clip_max(bb_5[3])-clip_min(bb_5[1])]
                bb_6   = [clip_min(bb_6[0])+256*1,clip_min(bb_6[1])+256*1,clip_max(bb_6[2])-clip_min(bb_6[0]),clip_max(bb_6[3])-clip_min(bb_6[1])]
                bb_7   = [clip_min(bb_7[0])+256*1,clip_min(bb_7[1])+256*2,clip_max(bb_7[2])-clip_min(bb_7[0]),clip_max(bb_7[3])-clip_min(bb_7[1])]
                bb_8   = [clip_min(bb_8[0])+256*1,clip_min(bb_8[1])+256*3,clip_max(bb_8[2])-clip_min(bb_8[0]),clip_max(bb_8[3])-clip_min(bb_8[1])]
                bb_9   = [clip_min(bb_9[0])+256*2,clip_min(bb_9[1])+256*0,clip_max(bb_9[2])-clip_min(bb_9[0]),clip_max(bb_9[3])-clip_min(bb_9[1])]
                bb_10  = [clip_min(bb_10[0])+256*2,clip_min(bb_10[1])+256*1,clip_max(bb_10[2])-clip_min(bb_10[0]),clip_max(bb_10[3])-clip_min(bb_10[1])]
                bb_11  = [clip_min(bb_11[0])+256*2,clip_min(bb_11[1])+256*2,clip_max(bb_11[2])-clip_min(bb_11[0]),clip_max(bb_11[3])-clip_min(bb_11[1])]
                bb_12  = [clip_min(bb_12[0])+256*2,clip_min(bb_12[1])+256*3,clip_max(bb_12[2])-clip_min(bb_12[0]),clip_max(bb_12[3])-clip_min(bb_12[1])]
                bb_13  = [clip_min(bb_13[0])+256*3,clip_min(bb_13[1])+256*0,clip_max(bb_13[2])-clip_min(bb_13[0]),clip_max(bb_13[3])-clip_min(bb_13[1])]
                bb_14  = [clip_min(bb_14[0])+256*3,clip_min(bb_14[1])+256*1,clip_max(bb_14[2])-clip_min(bb_14[0]),clip_max(bb_14[3])-clip_min(bb_14[1])]
                bb_15  = [clip_min(bb_15[0])+256*3,clip_min(bb_15[1])+256*2,clip_max(bb_15[2])-clip_min(bb_15[0]),clip_max(bb_15[3])-clip_min(bb_15[1])]
                bb_16  = [clip_min(bb_16[0])+256*3,clip_min(bb_16[1])+256*3,clip_max(bb_16[2])-clip_min(bb_16[0]),clip_max(bb_16[3])-clip_min(bb_16[1])]
         
                ff.write(' '.join(str(x) for x in bb_1 )+' '+str(id_1 )+'\n')
                ff.write(' '.join(str(x) for x in bb_2 )+' '+str(id_2 )+'\n')
                ff.write(' '.join(str(x) for x in bb_3 )+' '+str(id_3 )+'\n')
                ff.write(' '.join(str(x) for x in bb_4 )+' '+str(id_4 )+'\n')
                ff.write(' '.join(str(x) for x in bb_5 )+' '+str(id_5 )+'\n')
                ff.write(' '.join(str(x) for x in bb_6 )+' '+str(id_6 )+'\n')
                ff.write(' '.join(str(x) for x in bb_7 )+' '+str(id_7 )+'\n')
                ff.write(' '.join(str(x) for x in bb_8 )+' '+str(id_8 )+'\n')
                ff.write(' '.join(str(x) for x in bb_9 )+' '+str(id_9 )+'\n')
                ff.write(' '.join(str(x) for x in bb_10)+' '+str(id_10)+'\n')
                ff.write(' '.join(str(x) for x in bb_11)+' '+str(id_11)+'\n')
                ff.write(' '.join(str(x) for x in bb_12)+' '+str(id_12)+'\n')
                ff.write(' '.join(str(x) for x in bb_13)+' '+str(id_13)+'\n')
                ff.write(' '.join(str(x) for x in bb_14)+' '+str(id_14)+'\n')
                ff.write(' '.join(str(x) for x in bb_15)+' '+str(id_15)+'\n')
                ff.write(' '.join(str(x) for x in bb_16)+' '+str(id_16)+'\n')
        
        	cv2.imwrite('../../recognition_dataset_new/'+dataset+'/1/'+str(id)+'.jpg',new_img)

	
        ff.close()


f = open('face_recognition_dataset.pkl','rb')
data = pickle.load(f)
f.close()
face_ids = np.arange(0,len(data))
np.random.shuffle(face_ids)

low_train  = 0
high_train = int((len(data)*49)/800)
low_val    = high_train
high_val   = int(len(data)/16)


generate_recognition_dataset(dataset='train',low=low_train,high=high_train,data=data,face_ids=face_ids)
generate_recognition_dataset(dataset='val',  low=low_val  ,high=high_val,data=data,face_ids=face_ids)




