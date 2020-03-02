import glob
import numpy as np
import pandas as pd
import h5py
import os
import pickle
import os.path
from os import path

NUM_PACKET_THRESHOLD = 400 

def files2HDF5(local_path):
    # load the list of labled scanners
    IoTdict = {}
    with open(local_path+"tagged.txt") as IoT_file:
        for line in IoT_file:
            line = line.strip("\n")
            linesplit = line.split(',')
            if path.exists(local_path+"flows/"+linesplit[0]):
                IoTdict[linesplit[0]] = {"label":linesplit[1], "manufacturer":linesplit[2], "device_type":linesplit[3], "device_model":linesplit[4], "OS":linesplit[5]}

    # open a hdf5 file and create arrays
    shape = (len(IoTdict), NUM_PACKET_THRESHOLD, 24, 1) 
    hdf5_file = h5py.File(local_path+'dataset.hdf5', mode='w')
    hdf5_file.create_dataset("flows", shape, np.float32)


    dfObj = pd.DataFrame(columns=['label', 'manufacturer', 'device_type','device_model','OS'])

    for i,ip in enumerate(IoTdict):
        # print how many images are saved every 1000 flows
        if i % 1000 == 0 and i > 1:
            print('Train data: {}/{}'.format(i, shape[0]))
        # read an scan-packet-flow and shape it to (400, 24)

        df = pd.read_pickle(local_path+"flows/"+ip) 
        packet_flow = np.empty([NUM_PACKET_THRESHOLD, 24, 1], dtype = float)
        packet_flow[:,:,0] = df[['prtcl', 'tos', 'tot_len', 'ip_id', 'ttl', 'IPsrc_long', 'IPdst_long', 'srcPort', 'dstPort', 'tcp_seq', 'tcp_ack_seq', 'tcp_off', 'tcpdatalen', 'tcp_reserve', 'tcp_flag', 'tcp_win', 'tcp_urp', 'timestamp', 'TCP_OPT_NOP', 'TCP_OPT_MSS', 'TCP_OPT_WSCALE', 'TCP_OPT_SACKOK', 'TCP_OPT_SACK', 'TCP_OPT_TIMESTAMP']].to_numpy()
        dfObj = dfObj.append(IoTdict[ip], ignore_index=True)
        hdf5_file["flows"][i, ...] = packet_flow[None]
        
    hdf5_file.close()

    dfObj.to_hdf(local_path+'dataset.hdf5', 'labels', format='table', mode='r+')