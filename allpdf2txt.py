import os

# list of pdfs
pdf_dir = './pdfs'
txt_dir = './txts'
file_list = os.listdir(pdf_dir)[1:]
print(file_list)

# trim the space in pdf file(if any)
for idx,file_name in enumerate(file_list):
	if " " in file_name:
		after_name = file_name.replace(" ","_")
		os.rename(pdf_dir+'/'+file_name,pdf_dir+'/'+after_name)
		file_list[idx] = after_name
print(file_list)

# convert to txt
for file_addr in file_list:
	cmd = "pdf2txt.py pdfs/"+file_addr+"  > ./txts/"+file_addr[:-4]+".txt"
	print('running...')
	print(cmd)
	os.system(cmd)