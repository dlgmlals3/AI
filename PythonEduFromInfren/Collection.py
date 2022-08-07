def stringFunc():
    a = 'hellow world'
    b = "'hellow' world"
    c = '''Hello World'''
    d = """Hello World"""
    print(a, b, c, d)
    e = '''hello
    world'''
    print(e)

    a = 'Hello World' # 길이 : 11
    print(a[-1])
    # 문자열 슬라이싱
    a = 'Hello world'
    print(a[3:11])
    print(a[0:1])
    print(a[:5])

    a = 'hello world'
    # 대문자로 만듬.
    a.upper()

    # replace
    # 문자열 내의 특정문자를 치환
    a = 'hello world'
    a = a.replace ('h', 'j')
    print(a)

    # format
    # 문자열 내의 특정값을 변수로부터 초기화하여 동적으로 문자열 생성
    temperature = 10
    prob = 30
    a = '오늘 기온{}도 이고, 비올 확률은 {}% 입니다.'.format(temperature, prob)
    print(a)

    # split
    # 문자열을 특정한 문자로 구분하여 문자열 리스트로 치환
    a = 'hello world what a nice weather'
    print(a.split('w'))

#####################################
# list
#####################################
def ListFunc():
    # 순서가 있는 데이터들의 모임
    a = []
    print(a)
    a = [1, 2, 3, 5, 10]
    print(a)
    a = ['korea', 'canada', 1, 23, [34, 56]]
    print(len(a))

    a = 'hello world'
    b = list(a)
    print(b, len(b))

    a = 'hellow world nice weather'
    b = a.split()
    print(b)

    # indexing 문자열
    a = [1,2,3,4,5,6]
    print(a[2])
    print(a[5])

    # Replacing
    a = 'hello world'
    print(a[0]) # const 라서 못바꿈.. 새로 생성해야됌.
    b = 'jello world'
    c = 'j' + a[1:]
    print(b, c)
    print(a.replace('h', 'd'))

    # 리스트 슬라이싱
    a = [1, 2, 3, 4, 5]
    a[0] = 100
    a[-1] = 90
    print(a)

    # 리스트 slicig
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    print(a[4:7])
    print(a[:7])
    print(a[3:])

    # start:end:increment
    print(a[1:7:2])

    # 리스트 멤버 함수
    # append() : 리스트 끝에 항목을 추가함.
    a = [1, 2, 3, 4, 5]
    a.append(10)
    print(a)

    # extend() : 리스트를 연장
    a = [1, 2, 3, 4, 5,]
    b = [6, 7, 8, 9, 10]
    # [1, 2, 3, 4, ,5, 6, 7, 8, 9, 10]
    # a.append(b) # 리스트 자체가 추가됌.
    print(a)
    # a.extend(b)
    a += b
    print(a)

    # insert()
    # 리스트 원하는 위치에 추가 가능
    a = [1, 2, 3, 4, 5, 6]
    a.insert(1, 40)
    print(a)

    # remove
    # 값으로 항목 삭제
    a = [1, 2, 3, 4, 5]
    a.remove(2)
    print(a)

    # pop 지우고자 하는 인덱스 아이템 반환 후 삭제
    a = [1, 2, 3, 4, 5]
    d = a.pop(2)
    print(d, a)

    # index : 찾고자 하는 값의 인덱스를 반환
    a = [2, 6, 7, 9, 10]
    print(a.index(2))

    # in 키워드
    # 리스트 내에 해당 값이 존재하는지 확인
    a = [1, 2, 3, 4, 5, 10]
    b = 7
    print(b in a)

    # 리스트 정렬
    a = [9, 10, 7, 19, 1, 2, 20, 21, 7, 8]
    # a.sort(reverse=False)
    b = sorted(a)
    print('a: ', a)
    print('b: ', b)

#####################################
# Tuple
#####################################
def TupleFunc():
    # 복수개의 값을 갖는 컬렉션 타입
    a = [1, 2, 3]
    b = (1, 2, 3)
    a[0] = 100
    # b[0] = 100
    print('1aaa : ', type(a), a)
    print('2bbb : ', type(b), b)# item is not changed

    # tuple 많이 사용할때
    a, b, c, d = (100, 200, 300, 400)
    print(type(a))
    print(a, b)

    # tuple 사용하여 값 치환
    a = 5
    b = 4
    a, b = b, a
    print(a, b)
    # logic

#####################################
# Dictionary
#####################################
# 키와 값을 갖는 데이터 구조
# 키는 내부적으로 hash 값으로 저장.
# 순서를 따지지 않고 인덱스 없음.
def DictionaryFunc():
    a = {'Korea': 'Seoul',
         'Canada':'Ottawa',
         'USA':'Washington D.C'}
    b = {0:1, 1:6, 7:9, 8:10}
    print(type(a['Korea']), a['Korea'])
    # print(b[2]) # error
    a['Japan'] = 'Tokyo'
    a['Japan'] = 'Kyoto'
    a['China'] = 'Beijing'
    print(a)

    # update 두 딕셔너리를 병합함
    # 겹치는 키가 있다면 parameter로 전달되는 키 값이 overwrite 된다.
    a = {'a':1, 'b':2, 'c':3}
    b = {'a':2, 'd':4, 'e':5}
    a.update(b)

    # dicionary key 삭제
    # del 키워드 사용
    # pop 함수 사용
    a.pop('b')
    del a['a']
    print(a)
    c = 100
    del c
    # print(c)

    print(a)
    a.clear()
    print(a)

    # in
    # key 값 존재 확인
    a = {'a':1, 'b':2, 'c':3}
    b = [1,2,3,4,5,6,7,8,9,10, 100]

    print(a)
    print('b' in a)
    print('d' in a)
    # 리스트 vs dictionary 검색속도 빠름.
    print(100 in b)
    print('b' in a)

    # print(a['d']) # 프로그램 key error 발생
    print(a.get('d'))
    if 'd' in a:
        print(a['d'])

    print(a)
    print(list(a.keys()))
    print(list(a.values()))
    print(list(a.items()))


#####################################
# Set
#####################################
def SetFunc():
    a = {1, 1 ,2, 3, 3, 4, 1, 5}
    print(a)

    b = set(a)
    print(type(b), b)

    a = {1, 2, 3}
    b = {2, 3, 4}
    print(a.union(b)) # 합집합
    print(a.intersection(b)) # 교집합
    print(a.difference(b)) # 차집합
    print(a.issubset(b)) # 부분집합