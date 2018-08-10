# goban-image-reader
A tool for transforming goban image to sgf file using nerual network.

Usage(temporary version, in chinese):

0. 命令行输入 python --version, 确保使用的是python3
1. git clone https://github.com/chaossy/goban-image-reader.git
2. cd至项目根目录
3. (可选)解压缩input.tar.gz至根目录, 这个压缩包包含了500张左右我自己拍的照片和对应的sgf文件，用来创建之后的数据集
4. (可选)创建一个virtualenv, 在命令行输入 source activate your-virtualenv-name
5. pip install -r requirements.txt
6. 使用 python data_convert.py -syn -o -s path-to-your-sgf-dir 创建测试数据集。<br />
   选项说明：<br />
   -syn：从sgf文件生成合成棋盘图片用来创建测试数据集<br />
   -o: 输出合成的棋盘图片至 /project-root-dir/dataset/syn_image<br /> 
   -s: 输入的sgf文件夹，所有sgf必须放在根目录下<br />
   -i: 输入的图片文件夹，如果没有指定-syn，必须指定这个选项，可以使用之前从input.tar.gz解压的文件测试<br />
7. 使用 python main.py -es 测试前一步生成的数据集<br />
   选项说明：<br />
   -es: 测试合成数据集<br />
   -er: 测试真实数据集
8. 使用 python main.py -p filepath1 filepath2 预测输入的图片



TODO: finish this README file
