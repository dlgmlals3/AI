# 가변길이 함수 파라메터 앞에 별 붙이면됌 ==> 튜플로 인식
def test(*args):
    print(type(args))
    for item in args:
        print(item)

test(10)
test(10, 20, 30)

# 가변길이 키워드 파라메터 ==> 딕셔너리로 인식
def test2(**x):
    print(type(x))
    for key, value in x.items():
        print(key, ",", value, sep=' ')

test2(a=1, b=2, c=3, name='bob')


a = '오늘 온도: {}도, 강수확률은 : {}%'.format(20, 30)

# 가변길이 키워드 파라메터
b = '오늘 온도: {today_temp}도, 강수확률은 : {today_prob}% 내일 온도 : {tomoorow_temp}도'\
    .format(today_temp=20, today_prob=30, tomoorow_temp=23)
print(b)


# 람다 함수 lambda 함수
def square2(x):
    return x**2

square = lambda x:x**2
print(square(5))

def add2(x, y):
    return x + y
add2 = lambda x,y:x+y
print(add2(3, 4))

def str_len(s):
    return len(s)

strings = ['bob', 'charles', 'alexander3', 'teddy']
# strings.sort(key=str_len)
strings.sort(key=lambda s:len(s))
print(strings)

# filter : 특정 조건을 만족하는 요소만 남기고 필터링
# (함수, 리스트)
# map : 각 원소를 주어진 수식에 따라 변형하여 새로운 리스트를 반환
# reduce : 차례대로 앞 2개의 원소를 가지고 연산, 연산의 결과가 또 다음 연산의 입력으로 진행

def even(n):
    return n % 2 == 0


nums = [1, 2, 3, 6, 8, 9]
#print(list(filter(even, nums)))
print(list(filter(lambda n:n%2 == 0, nums)))

# map
print(list(map(lambda n:n**2, nums)))

# map
# print(list(map(even, nums)))
print(list(map(lambda n:n%2==0, nums)))

# reduce
import functools

a = [1, 3, 5, 8]
print(functools.reduce(lambda x, y:x + y, a))


# 입력 : 숫자 리스트
# 출력 : 숫자 리스트의 평균값

def mean(nums):
    return sum(nums) / len(nums)

print(mean([1, 2, 3]))
print(mean([1, 2, 3, 4, 5]))
print(mean([1,2, 3.0, 3.9, 8.7]))

# 소수 판별 (1과 자기 자신으로만 나눠지는 수)
# 입력 : 양의 정수 1개
# 출력 : boolean (소수 : True, 합성수:  False)
def is_prime(num):
    for i in range(2, num):
        if num % i == 0:
            return False
    return True

print(is_prime(100))
print(is_prime(89))
