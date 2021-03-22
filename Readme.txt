1.執行 main.py --query 加上想查詢文字 即可

2.文件檔都包含在壓縮檔裡，不用額外更改

3.cosine 使用util.py中的函式

4. Euclidean Distance: 將兩個vector裡的每一個數字相減然後平方，最後加總起來開根號，寫在util.py中
 Ex: (1,2,3) , (2,3,4)
 sqrt((1-2)^2 +(2-3)^2+ (3-4)^2) = sqrt(3)

5.idf計算：在VectorSpace build時會做計算，統計每個document Vector裡的數值，存在IDFVector中，只保存第一次，因為此值不會隨查詢輸入而更改

＊範例輸入：
python Vmain.py --query "drill wood sharp"