class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next


def my_code1(s):
    """有效的括号"""
    left = {
        '(': ')',
        '[': ']',
        '{': '}'
    }
    n = 0
    a = len(s)
    if a % 2 != 0 or s[0] in [']', '}', ')']:
        return False
    else:
        while s != '' and n < a/2:
            ss = list(s)
            l = len(s)
            for i in range(l-1):
                if left.get(ss[i]) == ss[i + 1]:
                    ss[i] = ''
                    ss[i+1] = ''
            s = ''.join(ss)
            n += 1
    return not s


def my_code2(s):
    """有效的括号-栈解法"""
    left = {
        ')': '(',
        ']': '[',
        '}': '{'
    }
    stack = []
    for i in list(s):
        if stack and i in left.keys():
            if stack[-1] == left.get(i):
                stack.pop()
            else:
                stack.append(i)
        else:
            stack.append(i)
    return not stack


def create_linked_list(values):
    dummy = ListNode()
    current = dummy
    for value in values:
        current.next = ListNode(value)
        current = current.next
    return dummy.next


def print_linked_list(head):
    current = head
    values = []
    while current:  # 遍历链表直到末尾
        values.append(current.value)  # 将当前节点的值加入列表
        current = current.next       # 移动到下一个节点
    print(" -> ".join(map(str, values)))


def my_code3(l1,l2):
    """合并有序链表"""
    current_l1 = l1
    current_l2 = l2
    head=None

    current = ListNode()
    print_linked_list(current)
    while current_l1 or current_l2:
        if current_l1.value <= current_l2.value:
            if not head:
                print(current_l1.value)
            current_l1 = current_l1.next
            if not current_l1.next:
                print(current_l2.next.value)
                break
        else:
            print(current_l2.value)

            current_l2 = current_l2.next
            if not current_l2.next:
                print(current_l1.next.value)
                break
    print_linked_list(current)


if __name__ == '__main__':
    # 有效的括号
    l1 = create_linked_list([1, 2, 3])
    l2 = create_linked_list([2, 3, 4])
    my_code3(l1, l2)
    # # 两数之和
    # a = [2,7,11,15]
    # target = 9
    # num = len(a)
    # for i in range(num):
    #     for j in range(num-i-1):
    #         if a[i] + a[i+j+1] == target:
    #             print([i, i+j+1])

    # # 罗马数字转整数
    # num_map = {
    #     'M': 1000,
    #     'D': 500,
    #     'C': 100,
    #     'L': 50,
    #     'X': 10,
    #     'V': 5,
    #     'I': 1
    # }
    # num = 123
    # re = []
    # for i, v in num_map.items():
    #     n = int(num/v)
    #     num = num - n*v
    #     for j in range(n):
    #         re.append(i)
    # result = ''.join(re)
    # print(result)

    # # 最长公共前缀
    # sorts = ['floww', 'flower', 'flow', 'ffa']
    # # sorts = ['dog', 'car', 'racecar']
    # min = len(sorts[0])
    # for i in sorts:
    #     l = len(i)
    #     if l < min:
    #         min = l
    # ww = ''
    # for j in range(min+1):
    #     num = sorts[0][:j]
    #     count = 0
    #     for word in sorts:
    #         if word.startswith(num):
    #             count += 1
    #     if count >= len(sorts):
    #         ww = num
    # return ww