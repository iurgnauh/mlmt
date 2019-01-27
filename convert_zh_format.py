
f_out = open("data/ontonotes/zh_train.txt","w")

#in chinese it is labeled as person not PER
label_list = ["O","B-ORG","I-ORG","B-PERSON","I-PERSON","B-LOC","I-LOC","B-GPE","I-GPE"]

with open("data/zh.ner.train","r") as f:
    for index, line in enumerate(f):
        if len(line)>2 and index<10:
            line = line.strip()
            data = line.split("\t")
            if data[-1] in label_list:
                print (data[0]+" "+data[-1]+"\n")
                f_out.write(data[0]+" "+data[-1]+"\n")
            else:
                begin = data[-1].split("-")[0]
                
                f_out.write(data[0]+" "+begin+"-MISC"+"\n")
        else:
            f_out.write("\n")
            #print("\n")