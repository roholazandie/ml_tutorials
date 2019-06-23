a = [1, 3, 4, 5, 6]

a_iterator = iter(a)

print(a_iterator.__next__())
print(a_iterator.__next__())


# import requests
# import re
#
#
# def get_pages(link):
#     links_to_visit = []
#     links_to_visit.append(link)
#     while links_to_visit:
#         current_link = links_to_visit.pop(0)
#         page = requests.get(current_link)
#         for url in re.findall('<a href="([^"]+)">', str(page.content)):
#             if url[0] == '/':
#                 url = current_link + url[1:]
#             pattern = re.compile('https?')
#             if pattern.match(url):
#                 links_to_visit.append(url)
#         yield current_link
#
#
# webpage = get_pages('http://www.bbc.com')
# for result in webpage:
#     print(result)

from itertools import count

counter = count(10)


# class MyInfiniteIterable():
#     def __init__(self, value):
#         self.value = value
#
#     def __next__(self):
#         return self.value
#
#     def __iter__(self):
#         return self
#
# m = MyInfiniteIterable(5)
# print(next(m))
# for item in m:
#     print(item)


# class FibonacciIterator:
#     def __init__(self):
#         self.prev = 0
#         self.curr = 1
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         value = self.curr
#         self.curr += self.prev
#         self.prev = value
#         return value
#
#
# fib = FibonacciIterator()


# for f in fib:
#     print(f)

#
# import requests
# import re
#
#
# def get_pages(link):
#     links_to_visit = []
#     links_to_visit.append(link)
#     while links_to_visit:
#         current_link = links_to_visit.pop(0)
#         page = requests.get(current_link)
#         for url in re.findall('<a href="([^"]+)">', str(page.content)):
#             if url[0] == '/':
#                 url = current_link + url[1:]
#             pattern = re.compile('https?')
#             if pattern.match(url):
#                 links_to_visit.append(url)
#         yield current_link
#
#
# webpage = get_pages('http://www.google.com')
# for result in webpage:
#     print(result)
#
# def firstn(n):
#     num = 0
#     while num < n:
#         yield num
#         num += 1
#
# f = firstn(10)
#
# print(f.__next__())
# print(f.__next__())
# print(f.__next__())



class MyRange():

    def __init__(self, start, end):
        self.current = start
        self.start = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current > self.end:
            raise StopIteration
        else:
            self.current += 1
            return self.current - 1


myrange = MyRange(3, 10)

for i in myrange:
    print(i)