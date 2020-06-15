form Files
	sentence inputDir E:\workspace\digit-training\data_train\
	sentence outputDir E:\workspace\digit-training\f0\output\
endform

writeInfoLine: "Hello"

appendInfoLine: "Thu muc dau vao: 'inputDir$'"
appendInfoLine: "Thu muc ket qua: 'outputDir$'"

createStr = Create Strings as file list... list 'inputDir$'*.wav

numberOfFiles = 800

appendInfoLine: "Tong so file Wav trong thu muc dau vao: 'numberOfFiles'"

for ifile to numberOfFiles
	# opens each file, one at a time
	select Strings list
	fileName$ = Get string... ifile
	clearinfo
	appendInfoLine: "####################################################"
	appendInfoLine: "'ifile'/'numberOfFiles'", tab$, "File name: 'fileName$'"
	@tinhthamsomotfile: "'fileName$'"	
endfor
# Clean all
removeObject: createStr

procedure tinhthamsomotfile: .fileName$
	
	# Tao ten file ket qua : a.wav ==> a.txt
	resultFileName$ = fileName$ - ".wav" + ".f0"
	resultFilePath$ = outputDir$ + resultFileName$
	appendInfoLine: "Output Filename: 'resultFilePath$'"

	
	
	# Doc file Wav
	Read from file... 'inputDir$''fileName$'
	Rename: "mywav"
	
	tmin = Get start time
	tmax = Get end time
	appendInfoLine: "Start time: 'tmin'"
	appendInfoLine: "End time: 'tmax'"
	
	nbFrame= 45
	shiftFrame = (tmax-tmin)/nbFrame
	
	duration = Get total duration
	appendInfoLine: "Duration: 'duration'"
	
	f0_minimum_m = 75
   	f0_maximum_m = 200
   	octave_jump = 0.3
   	#Voicing_threshold 0.65

	selectObject: "Sound mywav"	
	# Male Voices
	# appendInfoLine: "Gioi tinh: Male"
#	#To Pitch: 0, 75, 200	 
	To Pitch (cc)... 0 f0_minimum_m 15 no 0.03 0.45 octave_jump 0.35 0.14 f0_maximum_m
	
	appendInfoLine: "nbFrame: 'nbFrame'"
	#for i to (tmax-tmin)/shiftFrame
	for i from 1 to nbFrame
		time = tmin + i * shiftFrame
		pitch = Get value at time: time, "Hertz", "Linear"
		pitch[i] = pitch
		if pitch == undefined
			pitch[i] = 0
		endif

	endfor
	Remove

	removeObject: "Sound mywav"
	
	
	# Ghi thong tin vao file 
	# for i to (tmax-tmin)/shiftFrame
	appendInfoLine: "Writing Parameter File................."
	for i from 1 to nbFrame
		# Pitch----------------------------------------------
		strPitch$=string$(pitch[i])
		appendFileLine: "'resultFilePath$'",strPitch$
		
	endfor
	
endproc