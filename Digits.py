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


Digits_Golden = {"France_4_1_220v":[], 
              "France_6_1_234v":[],
              "France_64_7_068v":[],
              "Spain_364C_6_184v":[], 
              "Spain_423_4_300v":[],
              "Spain_423_5_374r":[],
              "Spain_423_6_384v":[]} 


Digits_Golden_placeholders = {"France_4_1_220v":[], 
              "France_6_1_234v":[],
              "France_64_7_068v":[],
              "Spain_364C_6_184v":[], 
              "Spain_423_4_300v":[],
              "Spain_423_5_374r":[],
              "Spain_423_6_384v":[]} 


Digits_Golden_insecure = {"France_4_1_220v":[], 
              "France_6_1_234v":[],
              "France_64_7_068v":[],
              "Spain_364C_6_184v":[], 
              "Spain_423_4_300v":[],
              "Spain_423_5_374r":[],
              "Spain_423_6_384v":[]}



Digits_Golden_set = set()



def get_golden(cipher, cipher_dict,cipher_dict_insecure, cipher_dict_place):
    untranscribed = 0 
    uncertain = 0
    for page in cipher_dict: 
        with open("GT/"+cipher+page+".txt", "r") as file:
            for line in file: 
                if re.search("[a-zA-Z][^SIGNATURE\s]", line) is not None: 
                    pass
                else:
                    line_list_placeholders = [] 
                    line_list_insecure = []
                    line_list = []
                    strip = line.strip('\n') 
                    no_paragraph = strip.split() 
                    for char in no_paragraph: 
                        if len(char) == 4 and char.isdigit(): 
                            pass
                        elif re.search(r'\d', char) and "?" in char:
                            uncertain += 1
                            line_list_insecure.append(char)
                            longs = list(char) 
                            longs.remove("?") 
                            joined = ''.join(longs)
                            line_list_placeholders.append(joined)
                            line_list.append(joined)
                            Digits_Golden_set.add(joined)
                        elif re.search(r'\d', char) and not "?" in char:
                            line_list_insecure.append(char)
                            line_list_placeholders.append(char)
                            line_list.append(char)
                            Digits_Golden_set.add(char)
                        elif char == " " or char == "SIGNATURE":
                            pass
                        elif char == "??":
                            untranscribed += 1
                            line_list_placeholders.append(char)
                            line_list_insecure.append(char)
                        elif char == "?":
                            line_list_placeholders.append(char)
                            line_list.append(char)
                            line_list_insecure.append(char)
                            uncertain += 1
                        else:
                            line_list_placeholders.append(char)
                            line_list.append(char)
                            Digits_Golden_set.add(char)
                    if not len(line_list) == 0:
                        cipher_dict_insecure[page].append(line_list_insecure)
                        cipher_dict_place[page].append(line_list_placeholders)
                        cipher_dict[page].append(line_list)


    print ("Symbols in test but not ground truth: ", untranscribed)
    print ("Uncertain cases: ", uncertain)




#-#-#-#-#-#-#-#-#-#-#-#-#
#-# Transcript  Tool  #-#
#-#-#-#-#-#-#-#-#-#-#-#-#

Digits_decrypt = {"France_4_1_220v":[], 
              "France_6_1_234v":[],
              "France_64_7_068v":[],
              "Spain_364C_6_184v":[], 
              "Spain_423_4_300v":[],
              "Spain_423_5_374r":[],
              "Spain_423_6_384v":[]} 

Digits_decrypt_placeholders = {"France_4_1_220v":[], 
              "France_6_1_234v":[],
              "France_64_7_068v":[],
              "Spain_364C_6_184v":[], 
              "Spain_423_4_300v":[],
              "Spain_423_5_374r":[],
              "Spain_423_6_384v":[]}

Digits_decrypt_omniglot = {"France_4_1_220v":[], 
              "France_6_1_234v":[],
              "France_64_7_068v":[],
              "Spain_364C_6_184v":[], 
              "Spain_423_4_300v":[],
              "Spain_423_5_374r":[],
              "Spain_423_6_384v":[]} 



Digits_decrypt_alphabet = set()



def decrypt_dicts(cipher_decrypt, folder, cipher_decrypt_place = None):
    missing_symbols = 0
    transcribed_symbols = 0
    untranscribed = 0
    

    for page in cipher_decrypt:
        with open("Decrypt/"+folder+"/"+page+".txt", "r") as file:
            for line in file:
                line_list = []
                line_list_placeholders = []
                strip = line.strip('\n')
                no_paragraph = strip.split()
                for char in no_paragraph:
                    if char == 'text':
                        pass
                    elif char == '?':
                        line_list.append(char)
                        missing_symbols += 1
                        if cipher_decrypt_place != None:
                            line_list_placeholders.append(char)
                    elif char == '??':
                        line_list_placeholders.append(char)
                        untranscribed += 1
                    else:
                        line_list.append(char)
                        Digits_decrypt_alphabet.add(char)
                        transcribed_symbols += 1
                        if cipher_decrypt_place != None:
                            line_list_placeholders.append(char)
                cipher_decrypt[page].append(line_list)
                if cipher_decrypt_place != None:
                    cipher_decrypt_place[page].append(line_list_placeholders)


    print("Missing symbols: ", str(missing_symbols)+" | " + str(round(missing_symbols/(transcribed_symbols+missing_symbols) *100,2))+"%")
    print("Untranscribed symbols: ", str(untranscribed)+" | "+ str(round(untranscribed/(transcribed_symbols+untranscribed+missing_symbols) *100,2))+ "%\n")






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
        for line1, line2 in itertools.zip_longest(cipher_dict[page1], golden_dict[page2], fillvalue=list()): 
            gold_len += len(line2) 
            for char in line1: 
                if char != "?":
                    cipher_len += 1
                    recognized += 1
        total += gold_len 
        print(page1+":", str(cipher_len)+"/"+str(gold_len), str(100*cipher_len/gold_len)+"%")
    print("\nTotal recognition =", str(round(recognized/total *100,2))+"% | ", str(recognized)+"/"+str(total))
    print("\nCharacters recognized:", str(len(alphabet))+"/"+str(len(gold_alphabet)), sorted(alphabet)) 
    print("\nFAILSAFE: Golden Alphabet=", sorted(gold_alphabet))
    print("\nAlphabet recognition: " , len(alphabet)*100 /len(gold_alphabet))
    print("\n - - Done - - -\n")





def evaluation(cipher_dict, golden_dict, insecure_dict = None):
    character_frequency = dict()
    golden_character_frequency = dict() 
    accuracy_num = 0
    accuracy_den = 0
    gold_insecurities_match = []
    test_insecurities_match = dict()
    common_insecurities = []
    placeholders_comp = dict()
    placeholders_gold = []
    for page1, page2 in zip(cipher_dict, golden_dict):
        for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]): 
            for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'): 
                character_frequency[char1] = character_frequency.get(char1, 0)+1 
                golden_character_frequency[char2] = golden_character_frequency.get(char2, 0)+1
                if char1 == char2:
                    true_positives[char1] = true_positives.get(char1, 0)+1
                    accuracy_num += 1
                if char1 == "??":
                    placeholders_comp[char2] = placeholders_comp.get(char2, 0)+1
                if char2 == "??":
                    placeholders_gold.append(tuple((char2,char1)))
    if insecure_dict:
        for page1, page2 in zip(cipher_dict, insecure_dict):
            for line1, line2 in zip(cipher_dict[page1], insecure_dict[page2]):
                for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                    if '?' in char2 and char2.split('?')[0] == char1:
                        common_insecurities.append(tuple((char1,char2)))
                    elif '?' in char2 and char2 != "??": 
                        gold_insecurities_match.append(tuple((char2,char1)))
                    elif char1 == '?':
                        test_insecurities_match[char2] = test_insecurities_match.get(char2, 0) +1

    for char in golden_character_frequency:
        accuracy_den += golden_character_frequency.get(char)
    print("### Model Accuracy ###")
    print(str(round(accuracy_num*100/accuracy_den,2))+"%")
    print("")
    print("### Model Error Rate ###")
    print(str(round((1-accuracy_num/accuracy_den)*100,2))+"%")
    print("")
    print("### Precision/Recall/F1 value for each character ###\n")

    mean_precision = 0
    mean_recall = 0
    count = 0

    for character in true_positives:
        precision = true_positives[character] / character_frequency[character] 
        recall = true_positives[character] / golden_character_frequency[character]

        mean_precision += precision
        mean_recall += recall
        count += 1
        print(character, "- P", round(precision, 4), "/ R", round(recall, 4), "/ Error Rate", round((1-precision), 4), "F1 scores", round(2*(recall*precision)/(recall+precision),2))
    mean_p = mean_precision*100/count 
    mean_r = mean_recall*100/count

    print("Mean Precision -", str(round(mean_p, 2))+"%")
    print("Mean Recall -", str(round(mean_r, 2))+"%")
    print("Avg error rate (chars) -", str(round(100-mean_p,2))+"%") 
    print("F1 score -", str(round(2*(mean_r*mean_p)/(mean_r+mean_p),2))+"%")

    print("\nCommon insecurities\n",common_insecurities)
    print("\nGold insecurities|Test matches\n",gold_insecurities_match)
    print("\nTest untranscribed|Gold matches\n",{k: v for k, v in sorted(test_insecurities_match.items(), key=lambda item: item[1], reverse = True)})

    print("\nPlaceholders in test set(symbols that were not recognized at all)\n", {k: v for k, v in sorted(placeholders_comp.items(), key=lambda item: item[1], reverse = True)})
    print("\nplaceholders-gold\n", placeholders_gold)

    
    # to select symbols for matrix
    sorted_true_positive_frequencies = ({k: v for k, v in sorted(true_positives.items(), key=lambda item: item[1], reverse = True)})
    sorted_golden_character_frequencies = {k: v for k, v in sorted(golden_character_frequency.items(), key=lambda item: item[1], reverse = True)}


    normalized_freq= dict()
    for key in true_positives:
        if key in golden_character_frequency: 
            normalized_freq[key] = true_positives[key]*100/golden_character_frequency[key]


    print("Normalized frequencies")
    print({k: v for k, v in sorted(normalized_freq.items(), key=lambda item: item[1], reverse = True)}) #sorted based on frequency
    print("\n - - Done - - -\n")



def ser(cipher_dict, golden_dict):
    tot_ser = 0
    tot_norm_ser = 0
    lines = 0
    for page1, page2 in zip(cipher_dict, golden_dict):
        for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
            truth = ''.join(line2) 
            if truth: 
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



def top_five_matrix_digits(cipher_dict, golden_dict, normalized = None):
    

    if normalized == True:
        keys = {"8" : "8", 
        "2" : "2",
        "4" : "4", 
        "3" : "3",
        "9" : "9",
        "6^." : "6^.",
        "?" : "?",
        "0_." : "0_.",
        "6" : "6",
        "5" : "5",
        "5^," : "5^,",
        "9_." : "9_.",
        "4_." : "4_.",
        "1" : "1",
        "??" : "??",
        "2_." : "2_.",
        "0" : "0",
        "8_." : "8_.",
        "8" : "8",
        "3^." : "3^."}

        actual = list()
        predicted = list()
        for page1, page2 in zip(cipher_dict, golden_dict):
            for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
                for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                    if char2 == '2' or char2 == '3' or char2 =='4' or char2 == '8' or char2 == '9':
                        actual.append(keys[char2])
                        if char1 == '2' or char1 == '3' or char1 =='4' or char1 == '8' or char1 == '9':
                            predicted.append(keys[char1])
                        elif char1 != '2' or char1 != '3' or char1 !='4' or char1 != '8' or char1 != "9":
                            predicted.append(keys[char1])


        print(unknown)
        actual.append("Other")
        predicted.append("Other")
        labels = [v for v in keys.values()]
        matrix = confusion_matrix(actual,predicted,labels = labels)
        frame = pd.DataFrame(matrix, index=labels, columns=labels)
    else:
        keys = {"0" : "0", 
        "2" : "2", 
        "3" : "3", 
        "8" : "8", 
        "5" : "5", 
        "Other" : "Other"}

        actual = list()
        predicted = list()
    
        for page1, page2 in zip(cipher_dict, golden_dict):
            for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
                for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                    if char2 == '2' or char2 == '3' or char2 =='8' or char2 == '0' or char2 == '5':
                        actual.append(keys[char2])
                        if char1 == '2' or char1 == '3' or char1 =='8' or char1 == '0' or char1 == '5':
                            predicted.append(keys[char1])
                        else:
                            predicted.append("Other")
        actual.append("Other")
        predicted.append("Other")
        labels = [i for i in keys.values()]
        matrix = confusion_matrix(actual,predicted,labels = labels)
        frame = pd.DataFrame(matrix, index=labels, columns=labels)
    frame.to_excel("output_best.xlsx")
    print("\n - - Matrix Done - - -\n")




def worst_five_matrix_digits(cipher_dict, golden_dict, normalized = None):

    unknown = set()
    if normalized == True:
        keys = {"4_." : "4_.",
        "0_." : "0_.",
        "1" : "1", 
        "7^'" : "7^'",
        "2_." : "2_.",
        "5^," : "5^,",
        "6^." : "6^.",
        "1_." : "1_.",
        "8_." : "8_.",
        "?" : "?", 
        "0" : "0",
        "3^." : "3^.",
        "0_." : "0_.",
        "2" : "2",
        "1^." : "1^.",
        "0-" : "0-",
        "5^." : "5^.",
        "2^." : "2^.",
        "4" : "4",
        "??" : "??",
        "9" : "9",
        "5" : "5",
        "3" : "3",
        "7^." : "7^."}


        actual = list()
        predicted = list()
        for page1, page2 in zip(cipher_dict, golden_dict):
            for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
                for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                    if char2 == '1' or char2 == '4_.' or char2 =='0_.' or char2 == '2_.' or char2 ==  "7^'":
                        actual.append(keys[char2])
                        if char1 == '1' or char1 == '4_.' or char1 =='0_.' or char1 == '2_.' or char1 == "7^'":
                            predicted.append(keys[char1])
                        elif char1 != '1' or char1 != '4_.' or char1 !='0_.' or char1 != '2_.' or char1 != "7^'":
                            predicted.append(keys[char1])

        labels = [v for v in keys.values()]
        matrix = confusion_matrix(actual,predicted,labels = labels)
        frame = pd.DataFrame(matrix, index=labels, columns=labels)
    else:
        keys = { "7^'" : "7^'",
        "0_." : "0_.",
        "6^." : "Six^.", 
        "3^." : "Three^.",
        "4_." : "Four_.", 
        "Other" : "Other"}

        actual = list()
        predicted = list()
        for page1, page2 in zip(cipher_dict, golden_dict):
            for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
                for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                    if char2 == '6^.' or char2 == '4_.' or char2 =='0_.' or char2 == '3^.' or char2 ==  "7^'":
                        actual.append(keys[char2])
                        if char1 == '6^.' or char1 == '4_.' or char1 =='0_.' or char1 == '3^.' or char1 == "7^'":
                            predicted.append(keys[char1])
                        else:
                            predicted.append("Other")

        actual.append("Other")
        predicted.append("Other")
        labels = [v for v in keys.values()]
        matrix = confusion_matrix(actual,predicted,labels = labels)
        frame = pd.DataFrame(matrix, index=labels, columns=labels)
    frame.to_excel("output_worst.xlsx")
    print("\n - - Matrix Done - - -\n")



def cross_digits(cipher_dict, golden_dict):

    keys = { "8": "8",
    "2" : "2",
    "3" : "3",
    "6" : "6",
    "4" : "4", 
    "5" : "5",
    "??" : "X",
    "5^,": "5^,",
    "9_.": "9_.",
    "4_.": "4_.",
    "0" :"0",
    "?" : "?",
    "1" : "1",
    "3^." : "3^.",
    "6_." : "6_.",
    "8_." : "8_.",
    "0_." : "0_.",
    "9" : "9",
    "2_." : "2_.",
    "6^." :"6^."
    }

    actual = list()
    predicted = list()
    for page1, page2 in zip(cipher_dict, golden_dict):
        for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
            for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                if char2 == '8' or char2 == '2' or char2 =='6' or char2 == '4' or char2 ==  "5"  or char2 ==  "3":
                    actual.append(keys[char2])
                    if char1 == '8' or char1 == '2' or char1 =='6' or char1 == '4' or char1 == "5" or char1 ==  "3":
                        predicted.append(keys[char1])
                    elif char1 != '8' or char1 != '2' or char1 !='6' or char1 != '4' or char1 != '5'  or char1 != '3':
                        predicted.append(keys[char1])
    labels = [v for v in keys.values()]
    matrix = confusion_matrix(actual,predicted,labels = labels)
    frame = pd.DataFrame(matrix, index=labels, columns=labels)

    frame.to_excel("output_cross.xlsx")
    print("\n - - Matrix Done - - -\n")



##############################
### FIRST ROUND OF TESTING ###
###      OPTIMIZATION      ###
##############################

get_golden("Digits_Golden/", Digits_Golden, Digits_Golden_insecure, Digits_Golden_placeholders)
decrypt_dicts(Digits_decrypt, "Digits_1shot_4trsh/")
ser(Digits_decrypt, Digits_Golden)
decrypt_dicts(Digits_decrypt, "Digits_1shot_6trsh/")
ser(Digits_decrypt, Digits_Golden)
decrypt_dicts(Digits_decrypt, "Digits_1shot_8trsh/")
ser(Digits_decrypt, Digits_Golden)
decrypt_dicts(Digits_decrypt, "Digits_5shot_4trsh/")
ser(Digits_decrypt, Digits_Golden)
decrypt_dicts(Digits_decrypt, "Digits_5shot_6trsh/")
ser(Digits_decrypt, Digits_Golden)
decrypt_dicts(Digits_decrypt, "Digits_5shot_8trsh/")
ser(Digits_decrypt, Digits_Golden)
decrypt_dicts(Digits_decrypt, "Digits_Omniglot")
ser(Digits_decrypt, Digits_Golden)



###############################
### SECOND ROUND OF TESTING ###
###############################

decrypt_dicts(Digits_decrypt, "Digits_5shot_4trsh/")
character_recognition(Digits_decrypt, Digits_Golden, Digits_decrypt_alphabet, Digits_Golden_set)
get_golden("Digits_Golden/placeholders/", Digits_Golden, Digits_Golden_insecure, Digits_Golden_placeholders)
decrypt_dicts(Digits_decrypt, "placeholders/Digits_5shot_4trsh/", Digits_decrypt_placeholders)
evaluation(Digits_decrypt_placeholders, Digits_Golden_placeholders, Digits_Golden_insecure)
top_five_matrix_digits(Digits_decrypt_placeholders, Digits_Golden_placeholders, False)
worst_five_matrix_digits(Digits_decrypt_placeholders, Digits_Golden_placeholders, False)
cross_digits(Digits_decrypt_placeholders, Digits_Golden_placeholders)
