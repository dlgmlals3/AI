def ConditionFunc():
    print("asdf")
    a = 10
    b = 8
    c = 11
    if a == 10 and b != 9 and c == 12:
        print('that is true')
    if not a == 10:
        print('a is ten')

    # False로 간주되는 값
    # None, 0, 0.0, '', [], (), {}, set()
    a = 0
    if a:
        print('print')
    a = [1, 2, 3]
    b = []
    if a:
        print('print list')

def ifFunc() :
    # 짝수인 경우 어떤 처리를 하고
    # 홀수인 경우 다른 처리를 해라
    a = 10
    if a % 2 == 0:
        print('짝수 ', a / 2)
    else:
        print('홀수', a + 1)

    a = 17
    if a % 4 == 0:
        print ('a is divisible by 4')
    elif a % 4 == 1:
        print ('a % 4 is 1')
    elif a % 4 == 2:
        print ('a % 4 is 2')
    else:
        print ('a % 4 is 3')

    a = 10
    b = 9
    c = 8

    if a == 10:
        if c == 8:
            if b == 8:
                print('a is ten and b is 8')
            else:
                print('a is ten and b is not 8')

    a = [1, 10, 9, 24, 566, 23, 45, 67, 89]

    i = 0
    while i < len(a):
        print('value : ' , a[i],' ', i)
        i += 1

    a = [1, 2, 4, 3, 5]
    for i in a :
        print(i, i * 2)

    a = [1, 2, 3, 4, 5]

    for number in a:
        print(number)

    a = 'hello world'
    for character in a:
        print(character)

    a = [1, 10, 3, 4, 5]
    for num in a:
        if num % 2 == 0:
            print(num)
        else:
            print(num+1)

    dica = {'korea':'seoul', 'japan':'tokyo', 'canada':'ottawa'}
    for k in dica:
        print(k, dica[k])

    for key in dica:
        print(key)

    for value in dica.values():
        print(value)

    for key, value in dica.items(): # tuple
        print(key, value)

    a = [1, 2, 4, 3, 5]
    # index와 값을 모두 사용가능.
    for index, val in enumerate(a):
        if index > 3:
            print (index, val)


    a = [100, 90, 80, 70, 60, 50]
    for num in a:
        if num < 80:
            break
        print(num)

    a = [100, 90, 80, 70, 60, 50]
    for num in a:
        if num >= 60 and num <= 70:
            continue
        print(num)

    # 1 ~ 100 리스트 생성
    l = list(range(1, 101))
    print(type(l), l)

    # 1 ~ 100 사이 5의 배수만을 갖는 리스트를 생성
    l = list(range(5, 101, 5))
    print(l)

#ConditionFunc()
#ifFunc()