#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-# LINE BY LINE EVAL #-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

import itertools
import editdistance
from sklearn.metrics import confusion_matrix
import re
import pandas as pd

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-# Golden standard dictionaries, lists for lines #-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


Ramanacoil_Golden = {"Ramanacoil_6958":[],
                     "Ramanacoil_6968":[],
                     "Ramanacoil_6973":[],
                     "Ramanacoil_6978":[],
                     "Ramanacoil_6983":[],
                     "Ramanacoil_6988":[],
                     "Ramanacoil_6993":[],
                     "Ramanacoil_7000":[]}


Ramanacoil_Golden_set = set()


def get_golden(cipher, cipher_dict):

    uncertain = 0

    for page in cipher_dict:
        with open("GT/"+cipher+page+".txt", "r") as file:
            for line in file:
                if "#COMMENTS" in line:
                    pass
                else:
                    line_list = []
                    strip = line.strip('\n')
                    no_paragraph = strip.split()
                    for char in no_paragraph:
                        if char == " ":
                            pass
                        elif char == "?":
                            line_list.append(char)
                            uncertain += 1
                        elif "?" and "/" in char:
                            x = re.split('\?|/', char)
                            Ramanacoil_Golden_set.add(x[0])
                            Ramanacoil_Golden_set.add(x[1])
                            line_list.append(tuple((x[0], x[1])))
                            uncertain += 1
                        elif "?" in char:
                            longs = list(char) 
                            longs.remove("?") 
                            joined = "".join(longs)
                            line_list.append(joined) #
                            Ramanacoil_Golden_set.add(joined)
                            uncertain += 1
                        elif re.search("[__]+", char):
                            line_list.append("_")
                            Ramanacoil_Golden_set.add("_")
                        else:
                            line_list.append(char)
                            Ramanacoil_Golden_set.add(char)
                    if not len(line_list) == 0:
                        cipher_dict[page].append(line_list)

                        

    print ("Uncertain cases: ", uncertain)



#-#-#-#-#-#-#-#-#-#-#-#-#
#-# Transcript  Tool  #-#
#-#-#-#-#-#-#-#-#-#-#-#-#


Ramanacoil_decrypt = {"Ramanacoil_6958":[],
                "Ramanacoil_6968":[],
                "Ramanacoil_6973":[],
                "Ramanacoil_6978":[],
                "Ramanacoil_6983":[],
                "Ramanacoil_6988":[],
                "Ramanacoil_6993":[],
                "Ramanacoil_7000":[]}

Ramanacoil_decrypt_alphabet = set()


def decrypt_dicts(cipher_decrypt, folder):
    missing_symbols = 0
    transcribed_symbols = 0

    for page in cipher_decrypt:
        with open("Decrypt/"+folder+"/"+page+".txt", "r") as file:
            for line in file:
                line_list = []
                strip = line.strip('\n')
                no_paragraph = strip.split()
                for char in no_paragraph:
                    if char == "?":
                        missing_symbols += 1
                        line_list.append(char)
                    elif char == "SPACE":
                        line_list.append("<SPACE>")
                        Ramanacoil_decrypt_alphabet.add("<SPACE>")
                        transcribed_symbols += 1
                    else:
                        line_list.append(char)
                        Ramanacoil_decrypt_alphabet.add(char)
                        transcribed_symbols += 1
                cipher_decrypt[page].append(line_list)

            

    print("Missing symbols: ", str(missing_symbols)+" | " + str(round(missing_symbols/(transcribed_symbols+missing_symbols) *100,2))+"%")



tuple_gold = {"Ramanacoil_6958":[],
                "Ramanacoil_6968":[],
                "Ramanacoil_6973":[],
                "Ramanacoil_6978":[],
                "Ramanacoil_6983":[],
                "Ramanacoil_6988":[],
                "Ramanacoil_6993":[],
                "Ramanacoil_7000":[]}


def tuples(golden_dict, cipher_dict, tuple_dict):
    for page in tuple_dict:
        for page1, page2 in zip(cipher_dict, golden_dict):
            for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
                line_list = []
                for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                    if type(char2) is tuple:
                        if char2[0] == char1 or char2[1] == char1:
                            line_list.append(char1)
                        else:
                            string = char2[0]+"/"+char2[1]
                            line_list.append(string)
                    else:
                        line_list.append(char2)
                if not len(line_list) == 0:
                    tuple_dict[page].append(line_list)



#-#-#-#-#-#-#-#-#-#
#-# EVALUATION  #-#
#-#-#-#-#-#-#-#-#-#

def character_recognition(cipher_dict, golden_dict, alphabet, gold_alphabet): 
    print("Character # recognized per page:")
    pages = 0
    total = 0
    recognized = 0
    for page1, page2 in zip(cipher_dict, golden_dict): 
        cipher_len = 0
        pages += 1
        gold_len = 0
        for line1, line2 in itertools.zip_longest(cipher_dict[page1], golden_dict[page2], fillvalue=list()): # 
            gold_len += len(line2) 
            for char in line1: 
                if char != "?":
                    cipher_len += 1
                    recognized += 1
        total += gold_len 
        print(page1+":", str(cipher_len)+"/"+str(gold_len), str(round(100*cipher_len/gold_len,2))+"%")
    print("\nTotal recognition =", str(round(recognized/total *100,2))+"% | ", str(recognized)+"/"+str(total))
    print("\nCharacters recognized:", str(len(alphabet))+"/"+str(len(gold_alphabet)), sorted(alphabet)) 
    print("\nFAILSAFE: Golden Alphabet=", sorted(gold_alphabet))
    print("\nAlphabet recognition: " , len(alphabet)*100 /len(gold_alphabet))
    print("\n - - Done - - -\n")




def ser(cipher_dict, golden_dict):
    tot_ser = 0
    tot_norm_ser = 0
    lines = 0
    for page1, page2 in zip(cipher_dict, golden_dict):
        for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
            truth = ''.join(line2) 
            if "?" in line1:
                unkn = 0
                for char in line1:
                    if char == "?":
                        unkn +=1
                for i in range(unkn):
                    line1.remove("?")
            pred = ''.join(line1) 
            SER = editdistance.eval(truth, pred) / len(truth) 
            norm_SER = editdistance.eval(truth, pred) / max(len(truth), len(pred)) 
            tot_ser += SER
            tot_norm_ser += norm_SER
            lines += 1
    avg_ser = tot_ser / lines 
    avg_norm_ser = tot_norm_ser / lines
    print("Avg. SER:", str(round(avg_ser*100,2))+"%") 
    print("Avg. SER, normalized:", str(round(avg_norm_ser*100,2))+"%")
    print("\n - - Done - - -\n")




def evaluation(cipher_dict, golden_dict):
    total_gold = dict() 
    total_test = dict()
    for page1, page2 in zip(cipher_dict, golden_dict):
        gold_symbols = dict() 
        test_symbols = dict()
        for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
            for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='untranscribed'): #
                test_symbols[char1] = test_symbols.get(char1, 0)+1
                gold_symbols[char2] = gold_symbols.get(char2, 0)+1
                total_gold[char2] = total_gold.get(char2, 0)+1
                total_test[char1] = total_test.get(char1, 0)+1

        sorted_gold_symbols = {k: v for k, v in sorted(gold_symbols.items(), key=lambda item: item[1], reverse = True)}
        sorted_test_symbols = {k: v for k, v in sorted(test_symbols.items(), key=lambda item: item[1], reverse = True)}


  
    sorted_total_gold = {k: v for k, v in sorted(total_gold.items(), key=lambda item: item[1], reverse = True)}
    sorted_total_test = {k: v for k, v in sorted(total_test.items(), key=lambda item: item[1], reverse = True)}



    df_gold = pd.DataFrame(list(sorted_total_gold.items()),columns = ['Symbol','Frequency'])
    df_test = pd.DataFrame(list(sorted_total_test.items()),columns = ['Symbol','Frequency'])
    df_test.to_excel("Ram_results.xlsx") 
    df_gold.to_excel(("Ram_gold_results.xlsx") )






##############################
### FIRST ROUND OF TESTING ###
###      OPTIMIZATION      ###
##############################



get_golden("Ramanacoil_Golden/", Ramanacoil_Golden)

# Borg model

decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Borg/Ramanacoil_1shot_4trsh")
tuples(Ramanacoil_Golden, Ramanacoil_decrypt, tuple_gold)
ser(Ramanacoil_decrypt, tuple_gold)
decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Borg/Ramanacoil_1shot_6trsh")
tuples(Ramanacoil_Golden, Ramanacoil_decrypt, tuple_gold)
ser(Ramanacoil_decrypt, tuple_gold)
decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Borg/Ramanacoil_5shot_4trsh")
tuples(Ramanacoil_Golden, Ramanacoil_decrypt, tuple_gold)
print(tuple_gold)
ser(Ramanacoil_decrypt, tuple_gold)
decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Borg/Ramanacoil_5shot_6trsh")
tuples(Ramanacoil_Golden, Ramanacoil_decrypt, tuple_gold)
ser(Ramanacoil_decrypt, tuple_gold)

# Copiale model
decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Copiale/Ramanacoil_1shot_4trsh")
tuples(Ramanacoil_Golden, Ramanacoil_decrypt, tuple_gold)
ser(Ramanacoil_decrypt, tuple_gold)
decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Copiale/Ramanacoil_1shot_6trsh")
tuples(Ramanacoil_Golden, Ramanacoil_decrypt, tuple_gold)
ser(Ramanacoil_decrypt, tuple_gold)
decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Copiale/Ramanacoil_5shot_4trsh")
tuples(Ramanacoil_Golden, Ramanacoil_decrypt, tuple_gold)
ser(Ramanacoil_decrypt, tuple_gold)
decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Copiale/Ramanacoil_5shot_6trsh")
tuples(Ramanacoil_Golden, Ramanacoil_decrypt, tuple_gold)
ser(Ramanacoil_decrypt, tuple_gold)

# Digits model
decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Digits/Ramanacoil_1shot_4trsh")
tuples(Ramanacoil_Golden, Ramanacoil_decrypt, tuple_gold)
ser(Ramanacoil_decrypt, tuple_gold)
decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Digits/Ramanacoil_1shot_6trsh")
tuples(Ramanacoil_Golden, Ramanacoil_decrypt, tuple_gold)
ser(Ramanacoil_decrypt, tuple_gold)
decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Digits/Ramanacoil_5shot_4trsh")
tuples(Ramanacoil_Golden, Ramanacoil_decrypt, tuple_gold)
ser(Ramanacoil_decrypt, tuple_gold)
decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Digits/Ramanacoil_5shot_6trsh")
tuples(Ramanacoil_Golden, Ramanacoil_decrypt, tuple_gold)
ser(Ramanacoil_decrypt, tuple_gold)

# Runic model
decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Runicus/Ramanacoil_1shot_4trsh")
tuples(Ramanacoil_Golden, Ramanacoil_decrypt, tuple_gold)
ser(Ramanacoil_decrypt, tuple_gold)
decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Runicus/Ramanacoil_1shot_6trsh")
tuples(Ramanacoil_Golden, Ramanacoil_decrypt, tuple_gold)
ser(Ramanacoil_decrypt, tuple_gold)
decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Runicus/Ramanacoil_5shot_4trsh")
tuples(Ramanacoil_Golden, Ramanacoil_decrypt, tuple_gold)
ser(Ramanacoil_decrypt, tuple_gold)
decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Runicus/Ramanacoil_5shot_6trsh")
tuples(Ramanacoil_Golden, Ramanacoil_decrypt, tuple_gold)
ser(Ramanacoil_decrypt, tuple_gold)

# Omniglot model
decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Omniglot/Ramanacoil_1shot_4trsh")
tuples(Ramanacoil_Golden, Ramanacoil_decrypt, tuple_gold)
ser(Ramanacoil_decrypt, tuple_gold)
decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Omniglot/Ramanacoil_1shot_6trsh")
tuples(Ramanacoil_Golden, Ramanacoil_decrypt, tuple_gold)
ser(Ramanacoil_decrypt, tuple_gold)
decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Omniglot/Ramanacoil_5shot_4trsh")
tuples(Ramanacoil_Golden, Ramanacoil_decrypt, tuple_gold)
ser(Ramanacoil_decrypt, tuple_gold)
decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Omniglot/Ramanacoil_5shot_6trsh")
tuples(Ramanacoil_Golden, Ramanacoil_decrypt, tuple_gold)
ser(Ramanacoil_decrypt, tuple_gold)


###############################
### SECOND ROUND OF TESTING ###
###############################

decrypt_dicts(Ramanacoil_decrypt, "Ramanacoil/Digits/Ramanacoil_5shot_4trsh")
character_recognition(Ramanacoil_decrypt, Ramanacoil_Golden, Ramanacoil_decrypt_alphabet, Ramanacoil_Golden_set)
evaluation(Ramanacoil_decrypt,Ramanacoil_Golden)
