import json
import os, sys
import cv2

def to_serializable(val):
    #Used by default
    return str(val)

def ts_float32(val):
    #Used if *val* is an instance of numpy.float32.
    return np.float64(val)

###############################################################
#This code reads the data from the good json file, and creates jsons to latter be processed into graphs
###############################################################
paths_good_data = []
file_dir = '/home/yonif/SimulAI/QATM/QATM_GOOD_JSON'
paths_good_data.extend([os.path.join(file_dir, i) for i in os.listdir(file_dir)])
all_results = []

for path in paths_good_data:
    print("-----------------------", path ,"-------------------------")
    with open(path, 'r') as file:
        count = 0
        data = file.read()
        arr_of_data = data.split("],")
        if str(arr_of_data[0])[0] == ",":
            arr_of_data[0] = str(arr_of_data[0][1:])
        arr_of_data_2 = []
        for i in range(len(arr_of_data)):
            if i==len(arr_of_data)-1:
                string = str(arr_of_data[i]).replace("(", "").replace(")", "")
                need_the_first = string.split("]")
                string = str(need_the_first[0]) + "]"
            else:
                string = str(arr_of_data[i]).replace("(", "").replace(")", "") + "]"
            try:
                arr_of_data_2.append(json.loads(string))
                count = count + 1
            except:
                print("Failed", i, " @@@@@ ", path)
        print("count = ", count, " all = ", len(arr_of_data))
        arr_of_data_2.sort(key=lambda elem: elem[0], reverse=True)
        os.makedirs("/home/yonif/SimulAI/QATM/QATM_RESULTS/" + os.path.basename(path)[:-4],
                    exist_ok=True)
        this_json_file = open("/home/yonif/SimulAI/QATM/QATM_JSONS/"+ os.path.basename(path)[:-4] + ".json", "w")
        this_results_array = []
        for i in range(len(arr_of_data_2)):
            this_results_array.append({"path": arr_of_data_2[i][1], "index": arr_of_data_2[i][3], "distance": arr_of_data_2[i][0], "window": [int(arr_of_data_2[i][2]), int(arr_of_data_2[i][3]), int(arr_of_data_2[i][2]) + int(arr_of_data_2[i][5]), int(arr_of_data_2[i][3]) + int(arr_of_data_2[i][4])]})

        json_obj = json.dumps(this_results_array, default=to_serializable)
        this_json_file.write(json_obj)
        this_json_file.close()
        all_results.append(this_results_array)
        print("DONE ---", path)

json_all_results = open("/home/yonif/SimulAI/QATM/all_results.json", "w")
json_obj = json.dumps({"data": all_results}, default=to_serializable)
json_all_results.write(json_obj)
json_all_results.close()

