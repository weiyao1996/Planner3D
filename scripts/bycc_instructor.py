# A Sentence will force the entire text to be considered as a single sentence, even if it has punctuation. 
# A Document will first tokenize the text into a list of sentences, and then annotate each sentence.

# from InstructorEmbedding import INSTRUCTOR
# model = INSTRUCTOR('hkunlp/instructor-base')
# sentence = "wardrobe front table, table left chair, chair stand on floor"
# instruction = "Represent the Science document for summarization:"
# embeddings = model.encode([[instruction,sentence]])
# print(embeddings)
# print(embeddings.shape) # (1,768)

# from sklearn.metrics.pairwise import cosine_similarity
# sentences_a = 'Represent the Science document for summarization: ','wardrobe front table, no chair'
# sentences_b = 'Represent the Science document for summarization: ','table left chair, chair stand on floor, wardrobe front table'
# embeddings_a = model.encode(sentences_a)
# embeddings_b = model.encode(sentences_b)
# similarities = cosine_similarity(embeddings_a,embeddings_b)
# print(similarities)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle
import numpy as np
from InstructorEmbedding import INSTRUCTOR
from shutil import move

# model_instructor = INSTRUCTOR('hkunlp/instructor-base').cuda()
# instruction = "Represent the Science document for summarization:"
# CLIP_path = "/home/weiy1/code/commonscenes/Data/visualization_supp/"
# LLM_path = "/home/weiy1/code/commonscenes/Data/visualization2/"
# scans = os.listdir(CLIP_path)
# for scan in scans:
#     print("Working on "+scan)
#     clip_feats_path = CLIP_path + scan + '/CLIP_small_' + scan + '.pkl'
#     clip_w_llm_feats_path = LLM_path + scan
#     if os.path.exists(clip_feats_path):
#         clip_feats_dic = pickle.load(open(clip_feats_path, 'rb'))
#         clip_feats_ins = clip_feats_dic['instance_feats']
#         clip_feats_rel = clip_feats_dic['rel_feats']
#         clip_feats_order = clip_feats_dic['instance_order']
#         sentences = ', '.join(clip_feats_rel.keys())
#         llm_embed = model_instructor.encode([[instruction,sentences]])
#         llm_embed_ins = np.tile(llm_embed, (clip_feats_ins.shape[0], 1)) # (num_instance+1, 768)
#         llm_embed_rel = np.squeeze(llm_embed) # (768)
#         clip_feats_ins = np.concatenate((clip_feats_ins, llm_embed_ins), axis=1) # (num_instance+1, 512+768)
#         for key in clip_feats_rel.keys():
#             clip_feats_rel[key] = np.concatenate((clip_feats_rel[key], llm_embed_rel))
#         if not os.path.exists(clip_w_llm_feats_path):
#             os.makedirs(clip_w_llm_feats_path)
#         new_dict = {'instance_feats': clip_feats_ins, 'rel_feats': clip_feats_rel, 'instance_order': clip_feats_order}
#         with open(clip_w_llm_feats_path + '/CLIP_small_' + scan + '.pkl', 'wb') as fp:
#             pickle.dump(new_dict, fp)

# from transformers import GPT2Tokenizer, GPT2Model

import openai
openai.api_key = "sk-yrXRlZ3M6j421f7cRt5SSHS4mQBKwjnzQ24NDOrn5ThM127R"

completion = openai.ChatCompletion.create(
  model="gpt-4",
  temperature=0,
  top_p=0,
  messages=[
    {"role": "system", "content": "You are a 3D indoor scene designer."},
    {"role": "system", "content": "Please summarize the following sentences. There are a double bed and two nightstand. The nightstand is smaller than the bed."} 
  ]
)

print(completion.choices[0].message.content)

# def form_prompt_for_chatgpt(text_input, top_k, stats, supporting_examples,
#                             train_features=None, val_feature=None):
#     message_list = []
#     unit_name = 'pixel' if args.unit in ['px', ''] else 'meters'
#     class_freq = [f"{obj}: {round(stats['class_frequencies'][obj], 4)}" for obj in stats['object_types']]
#     rtn_prompt = 'You are a 3D indoor scene designer. \nInstruction: synthesize the 3D layout of an indoor scene. ' \
#                 'The generated 3D layout should follow the CSS style, where each line starts with the furniture category ' \
#                 'and is followed by the 3D size, orientation and absolute position. ' \
#                 "Formally, each line should follow the template: \n" \
#                 f"FURNITURE {{length: ?{args.unit}: width: ?{args.unit}; height: ?{args.unit}; orientation: ? degrees; left: ?{args.unit}; top: ?{args.unit}; depth: ?{args.unit};}}\n" \
#                 f'All values are in {unit_name} but the orientation angle is in degrees.\n\n' \
#                 f"Available furnitures: {', '.join(stats['object_types'])} \n" \
#                 f"Overall furniture frequencies: ({'; '.join(class_freq)})\n\n"

#     message_list.append({'role': 'system', 'content': rtn_prompt})
#     last_example = f'{text_input[0]}Layout:\n'
#     total_length = len(tokenizer(rtn_prompt + last_example)['input_ids'])


#     if args.icl_type == 'k-similar':
#         assert train_features is not None
#         sorted_ids = get_closest_room(train_features, val_feature)
#         supporting_examples = [supporting_examples[id] for id in sorted_ids[:top_k]]
#         if args.test:
#             print("retrieved examples:")
#             print("\n".join(sorted_ids[:top_k]))

#     # loop through the related supporting examples, check if the prompt length exceed limit
#     for i, supporting_example in enumerate(supporting_examples[:top_k]):
#         cur_len = len(tokenizer(supporting_example[0]+supporting_example[1])['input_ids'])
#         if total_length + cur_len > args.gpt_input_length_limit:  # won't take the input that is too long
#             print(f"{i+1}th exemplar exceed max length")
#             break
#         total_length += cur_len

#         current_messages = [
#             {'role': 'user', 'content': supporting_example[0]+"Layout:\n"},
#             {'role': 'assistant', 'content': supporting_example[1].lstrip("Layout:\n")},
#         ]
#         message_list = message_list + current_messages
    
#     # concatename prompts for gpt4
#     message_list.append({'role': 'user', 'content': last_example})

#     return message_list

# prompt_for_gpt3 = form_prompt_for_chatgpt(
#                     text_input=val_example,
#                     top_k=top_k,
#                     stats=stats,
#                     supporting_examples=supporting_examples,
#                     train_features=train_features,
#                     val_feature=val_features[val_id]
#                 )

response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages="You are a 3D indoor scene designer. Please summarize the following sentences. There are a double bed and two nightstand. The nightstand is smaller than the bed.",#prompt_for_gpt3,
                        temperature=0.7,
                        max_tokens=1024,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        stop="Condition:",
                        n=1,
                    )

tokenizer_gpt4 = GPT2Tokenizer.from_pretrained("gpt4")
model_gpt4 = GPT4ForSequenceClassification.from_from_pretraind("gpt4")
# CLIP_path = "/home/weiy1/code/commonscenes/Data/visualization/"
# LLM_path = "/home/weiy1/code/commonscenes/Data/visualization_gpt/"
# scans = os.listdir(CLIP_path)
# for scan in scans:
#     print("Working on "+scan)
#     clip_feats_path = CLIP_path + scan + '/CLIP_small_' + scan + '.pkl'
#     clip_w_llm_feats_path = LLM_path + scan
#     if os.path.exists(clip_feats_path):
#         clip_feats_dic = pickle.load(open(clip_feats_path, 'rb'))
#         clip_feats_ins = clip_feats_dic['instance_feats']
#         clip_feats_rel = clip_feats_dic['rel_feats']
#         clip_feats_order = clip_feats_dic['instance_order']
#         sentences = ', '.join(clip_feats_rel.keys())
#         llm_embed = tokenizer_gpt4.encode(sentences, return_tensors="pt")
#         output_gpt4 = model_gpt4(llm_embed)
#         llm_embed_ins = np.tile(llm_embed, (clip_feats_ins.shape[0], 1)) # (num_instance+1, 768)
#         llm_embed_rel = np.squeeze(llm_embed) # (768)
#         clip_feats_ins = np.concatenate((clip_feats_ins, llm_embed_ins), axis=1) # (num_instance+1, 512+768)
#         for key in clip_feats_rel.keys():
#             clip_feats_rel[key] = np.concatenate((clip_feats_rel[key], llm_embed_rel))
#         if not os.path.exists(clip_w_llm_feats_path):
#             os.makedirs(clip_w_llm_feats_path)
#         new_dict = {'instance_feats': clip_feats_ins, 'rel_feats': clip_feats_rel, 'instance_order': clip_feats_order}
#         with open(clip_w_llm_feats_path + '/CLIP_small_' + scan + '.pkl', 'wb') as fp:
#             pickle.dump(new_dict, fp)

## for processing test data
# LLM_path = "/home/weiy1/code/commonscenes/Data/visualization_train/"
# dst_path = "/home/weiy1/code/commonscenes/Data/visualization1/"
# scans = os.listdir(dst_path)
# for scan in scans:
#     clip_w_llm_feats_path = dst_path + scan + '/CLIP_small_' + scan + '.pkl'
#     if os.path.exists(clip_w_llm_feats_path):
#         clip_feats_dic = pickle.load(open(clip_w_llm_feats_path, 'rb'))
#         clip_feats_ins = clip_feats_dic['instance_feats']
#         clip_feats_rel = clip_feats_dic['rel_feats']
#         if clip_feats_ins.shape[1] != 1280:
#             print("Working on "+scan)
                # if not os.path.exists(dst_path + scan):
                #     os.makedirs(dst_path + scan)
                # move(clip_w_llm_feats_path, dst_path + scan + '/CLIP_small_' + scan + '.pkl')
        # for key in clip_feats_rel.keys():
        #     if clip_feats_rel[key].shape[0] != 1280:
        #         print(scan + "error!@-@ ")
        #         print(clip_feats_rel[key].shape)
        #         # break