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

Runicus_Golden = {"page30":[], 
              "page97":[],
              "page99":[],
              "page173":[]}

Runicus_Golden_placeholders = {"page30":[], 
              "page97":[],
              "page99":[],
              "page173":[]}

Runicus_Golden_insecure = {"page30":[], 
              "page97":[],
              "page99":[],
              "page173":[]}


Runicus_Golden_set = set() 



def get_golden(cipher, cipher_dict, cipher_dict_insecure, cipher_dict_place):
    untranscribed = 0 
    uncertain = 0
    for page in cipher_dict: 
        with open("GT/"+cipher+page+".txt", "r") as file:
            for line in file: 
                line_list_placeholders = []
                line_list_insecure = []
                line_list = [] 
                strip = line.strip('\n') 
                clear = re.sub(r'<[^<>]*>', '', strip)
                for char in clear: 
                    if char == " ": 
                        pass
                    elif char == "?": 
                        last = line_list_insecure[-1]
                        del line_list_insecure[-1]
                        line_list_insecure.append(last+char)
                        uncertain += 1
                    elif char == "X":
                        untranscribed += 1
                        line_list_placeholders.append(char)
                        line_list_insecure.append(char)
                    else:
                        line_list_placeholders.append(char)
                        line_list.append(char)
                        line_list_insecure.append(char)
                        Runicus_Golden_set.add(char)    
                cipher_dict[page].append(line_list)
                cipher_dict_place[page].append(line_list_placeholders)
                cipher_dict_insecure[page].append(line_list_insecure)
    print ("Symbols in test but not in ground truth: ", untranscribed)
    print ("Uncertain cases: ", uncertain)




#-#-#-#-#-#-#-#-#-#-#-#-#
#-# Transcript  Tool  #-#
#-#-#-#-#-#-#-#-#-#-#-#-#

Runicus_decrypt = {"page30":[], 
              "page97":[],
              "page99":[],
              "page173":[]} 

Runicus_decrypt_placeholders = {"page30":[], 
              "page97":[],
              "page99":[],
              "page173":[]} 



Runicus_decrypt_alphabet = set()



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
                    if char == "?":
                        missing_symbols += 1
                        line_list.append(char)
                        if cipher_decrypt_place != None:
                            line_list_placeholders.append(char)
                    elif char == "X":
                        line_list_placeholders.append(char)
                        untranscribed += 1
                    else:
                        line_list.append(char)
                        Runicus_decrypt_alphabet.add(char)
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

def character_recognition(cipher_dict, golden_dict, alphabet, gold_alphabet): #alphabet:the tested one
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



def evaluation(cipher_dict, golden_dict, insecure_dict = None):
    character_frequency = dict()
    golden_character_frequency = dict() 
    accuracy_num = 0
    accuracy_den = 0
    gold_insecurities_match = []
    test_insecurities_match = dict()
    common_insecurities = []
    placeholders_comp = dict()
    placeholders_gold = dict()
    for page1, page2 in zip(cipher_dict, golden_dict):
        for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]): 
            for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'): 
                character_frequency[char1] = character_frequency.get(char1, 0)+1 
                golden_character_frequency[char2] = golden_character_frequency.get(char2, 0)+1
                if char1 == char2:
                    true_positives[char1] = true_positives.get(char1, 0)+1
                    accuracy_num += 1
                if char1 == "X": 
                    placeholders_comp[char2] = placeholders_comp.get(char2, 0)+1 
                if char2 == "X":
                    placeholders_gold[char1] = placeholders_gold.get(char1,0)+1
    if insecure_dict:
        for page1, page2 in zip(cipher_dict, insecure_dict):
            for line1, line2 in zip(cipher_dict[page1], insecure_dict[page2]):
                for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                    if '?' in char2 and char2.split('?')[0] == char1:
                        common_insecurities.append(tuple((char1,char2)))
                    elif '?' in char2: 
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
        print(character, "- P", round(precision, 4), "/ R", round(recall, 4), "/ Error Rate", round((1-precision), 4),"F1 scores", round(2*(recall*precision)/(recall+precision),2))
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
    print("\nPlaceholders in gold set(symbols existing in test transcription but not in gold set)\n", {k: v for k, v in sorted(placeholders_gold.items(), key=lambda item: item[1], reverse = True)}, "\n")

    
    # to select symbols for matrix
    sorted_true_positive_frequencies = ({k: v for k, v in sorted(true_positives.items(), key=lambda item: item[1], reverse = True)})
    sorted_golden_character_frequencies = {k: v for k, v in sorted(golden_character_frequency.items(), key=lambda item: item[1], reverse = True)}

    normalized_freq= dict()
    for key in true_positives:
        if key in golden_character_frequency: 
            normalized_freq[key] = true_positives[key]*100/golden_character_frequency[key]

    print("True positives frequencies:")
    print(sorted_true_positive_frequencies)
    print("\n")
    print("Gold frequencies:")
    print(sorted_golden_character_frequencies)
    print("\n")
    print("Normalized frequencies:")
    print({k: v for k, v in sorted(normalized_freq.items(), key=lambda item: item[1], reverse = True)})


     # ________________________________________________________
    # Find the types of symbols that constitute this dataset |
    # ________________________________________________________

    print("All gold symbols ", accuracy_den)
    print(sorted_golden_character_frequencies)
    print(character_frequency)
    punc = 0
    for key,value in sorted_golden_character_frequencies.items():
        if key == ":" or key == ";":
            punc += value
    total_punc = punc/1570 # 1570 is accuracy_den - symbols in test but not in ground truth
    print("Punctuation: " ,total_punc)
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
    avg_ser = tot_ser / lines #mean value of all pages
    avg_norm_ser = tot_norm_ser / lines
    print("Avg. SER:", str(round(avg_ser*100,2))+"%") 
    print("Avg. SER, normalized:", str(round(avg_norm_ser*100,2))+"%")
    print("\n - - Done - - -\n")





def top_five_matrix_runic(cipher_dict, golden_dict, normalized = None):

    if normalized == True:
        keys = {"o" : "ᚮ", 
        "n" : "ᚿ",
        "k" : "ᚴ ", 
        "u" : "ᚢ",
        "t" : "ᛐ",
        "d" : "ᛑ ",
        "l" : "ᛚ",
        "r" : "ᚱ",
        "g" : "ᚵ ",
        "b" : "ᛒ",
        "i" : "ᛁ"}

        actual = list()
        predicted = list()
        for page1, page2 in zip(cipher_dict, golden_dict):
            for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
                for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                    if char2 == 'o' or char2 == 'n' or char2 =='k' or char2 == 'u' or char2 == 't':
                        actual.append(keys[char2])
                        if char1 == 'o' or char1 == 'n' or char1 =='k' or char1 == 'u' or char1 == 't':
                            predicted.append(keys[char1])
                        elif char1 != 'o' or char1 != 'n' or char1 !='k' or char1 != 'u' or char1 != "t":
                            predicted.append(keys[char1])


        labels = [v for v in keys.values()]
        matrix = confusion_matrix(actual,predicted,labels = labels)
        frame = pd.DataFrame(matrix, index=labels, columns=labels)
    else:
        keys = {":" : "colon", 
        "æ" : "ᛅ", 
        "a" : "ᛆ", 
        "n" : "ᚿ", 
        "r" : "ᚱ", 
        "Other" : "Other"}

        actual = list()
        predicted = list()
    
        for page1, page2 in zip(cipher_dict, golden_dict):
            for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
                for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                    if char2 == ':' or char2 == 'æ' or char2 =='a' or char2 == 'n' or char2 == 'r':
                        actual.append(keys[char2])
                        if char1 == ':' or char1 == 'æ' or char1 =='a' or char1 == 'n' or char1 == 'r':
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




def worst_five_matrix_runic(cipher_dict, golden_dict, normalized = None):


    if normalized == True:
        keys = {";" : ";", 
        "z" : "ᛎ",
        "e" : "ᚽ", 
        ":" : ":",
        "æ" : "ᛅ",
        "d" : "ᛑ",
        "þ" : "ᚦ",
        "ø" : "ᚯ",
        "h" : "ᛡ ",
        "g" : "ᚵ",
        "b" : "ᛒ",
        "l" : "ᛚ",
        "n" : "ᚿ",
        "o" : "ᚮ",
        "X" : "X",
        "s" : "ᛋ",
        "r" : "ᚱ",
        "?" : "?"}

        actual = list()
        predicted = list()
        for page1, page2 in zip(cipher_dict, golden_dict):
            for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
                for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                    if char2 == ';' or char2 == 'z' or char2 =='e' or char2 == 'æ' or char2 ==  ":":
                        actual.append(keys[char2])
                        if char1 == ';' or char1 == 'z' or char1 =='e' or char1 == 'æ' or char1 == ":":
                            predicted.append(keys[char1])
                        elif char1 != ';' or char1 != 'z' or char1 !='e' or char1 != 'æ' or char1 != ":":
                            predicted.append(keys[char1])
        labels = [v for v in keys.values()]
        matrix = confusion_matrix(actual,predicted,labels = labels)
        frame = pd.DataFrame(matrix, index=labels, columns=labels)
    else:
        keys = {"p" : "ᛔ", 
        "z" : "ᛎ",
        "y" : "ᛦ", 
        "b" : "ᛒ",
        "ø" : "ᚯ",
        "Other" : "Other"}

        actual = list()
        predicted = list()
        for page1, page2 in zip(cipher_dict, golden_dict):
            for line1, line2 in zip(cipher_dict[page1], golden_dict[page2]):
                for char1, char2 in itertools.zip_longest(line1, line2, fillvalue='-'):
                    if char2 == 'p' or char2 == 'z' or char2 =='y' or char2 == 'ø' or char2 ==  "b":
                        actual.append(keys[char2])
                        if char1 == 'p' or char1 == 'z' or char1 =='y' or char1 == 'ø' or char1 == "b":
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


##############################
### FIRST ROUND OF TESTING ###
###      OPTIMIZATION      ###
##############################

get_golden("Runicus_Golden/", Runicus_Golden, Runicus_Golden_insecure, Runicus_Golden_placeholders)
decrypt_dicts(Runicus_decrypt, "Runicus_1shot_4trsh/")
ser(Runicus_decrypt, Runicus_Golden)
decrypt_dicts(Runicus_decrypt, "Runicus_1shot_6trsh/")
ser(Runicus_decrypt, Runicus_Golden)
decrypt_dicts(Runicus_decrypt, "Runicus_1shot_8trsh/")
ser(Runicus_decrypt, Runicus_Golden)
decrypt_dicts(Runicus_decrypt, "Runicus_5shot_4trsh/")
ser(Runicus_decrypt, Runicus_Golden)
decrypt_dicts(Runicus_decrypt, "Runicus_5shot_6trsh/")
ser(Runicus_decrypt, Runicus_Golden)
decrypt_dicts(Runicus_decrypt, "Runicus_5shot_8trsh/")
ser(Runicus_decrypt, Runicus_Golden)
decrypt_dicts(Runicus_decrypt, "Runicus_Omniglot")
ser(Runicus_decrypt, Runicus_Golden)



###############################
### SECOND ROUND OF TESTING ###
###############################

decrypt_dicts(Runicus_decrypt, "Runicus_5shot_4trsh/")
character_recognition(Runicus_decrypt, Runicus_Golden, Runicus_decrypt_alphabet, Runicus_Golden_set)
get_golden("Runicus_Golden/placeholders/", Runicus_Golden, Runicus_Golden_insecure, Runicus_Golden_placeholders)
decrypt_dicts(Runicus_decrypt, "placeholders/Runicus_5shot_4trsh/", Runicus_decrypt_placeholders)
evaluation(Runicus_decrypt_placeholders, Runicus_Golden_placeholders, Runicus_Golden_insecure)
top_five_matrix_runic(Runicus_decrypt_placeholders, Runicus_Golden_placeholders, False)
worst_five_matrix_runic(Runicus_decrypt_placeholders, Runicus_Golden_placeholders, False)

