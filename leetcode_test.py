class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next


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


def my_code_1(a, target):
    """两数之和"""
    num = len(a)
    for i in range(num):
        for j in range(num-i-1):
            if a[i] + a[i+j+1] == target:
                return [i, i+j+1]


def my_code_9(x):
    """回文数"""
    if x < 0:
        return False
    else:
        num = list(map(int, str(x)))
        l = len(num)
        y = []
        for i in range(l):
            y.append(str(num[l - 1 - i]))
        s = "".join(y)
        if int(s) == x:
            return True
        else:
            return False


def my_code_13(num):
    """罗马数字转整数"""
    num_map = {
        'M': 1000,
        'D': 500,
        'C': 100,
        'L': 50,
        'X': 10,
        'V': 5,
        'I': 1
    }
    re = []
    for i, v in num_map.items():
        n = int(num/v)
        num = num - n*v
        for j in range(n):
            re.append(i)
    result = ''.join(re)
    return result


def my_code_14(sorts):
    """最长公共前缀"""
    min = len(sorts[0])
    for i in sorts:
        l = len(i)
        if l < min:
            min = l
    ww = ''
    for j in range(min+1):
        num = sorts[0][:j]
        count = 0
        for word in sorts:
            if word.startswith(num):
                count += 1
        if count >= len(sorts):
            ww = num
    return ww


def my_code_20_1(s):
    """有效的括号-硬解"""
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


def my_code_20_2(s):
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


def my_code_21(l1,l2):
    """合并有序链表"""
    current_l1 = l1
    current_l2 = l2
    current = ListNode()
    head = current
    while current_l1 or current_l2:
        print_linked_list(current_l1)
        print_linked_list(current_l2)
        if not current_l1:
            current.next = current_l2
            break
        elif not current_l2:
            current.next = current_l1
            break
        elif current_l1.value <= current_l2.value:
            current.next = current_l1
            current_l1 = current_l1.next
        else:
            current.next = current_l2
            current_l2 = current_l2.next
        current = current.next
        print_linked_list(current)
    return head.next


def my_code_26(nums):
    """删除有序数组中的重复项"""
    i = 0
    l = len(nums)
    if l <=1:
        return l
    else:
        while i <= l-1:
            if nums[i] == nums[i+1]:
                del nums[i+1]
            else:
                i += 1
            if i == len(nums) - 1:
                break
    return len(nums),nums


def my_code_27(nums, val):
    """移除元素"""
    i = 0
    l = len(nums)
    while i <= l-1:
        if nums[i] == val:
            del nums[i]
        else:
            i += 1
        if i == len(nums):
            break
    return len(nums),nums\



def my_code_28(haystack, needle):
    """找出字符串中第一个匹配项的下标"""
    n = -1
    if len(needle)>len(haystack):
        return -1
    else:
        for i,v in enumerate(haystack):
            if v == needle[0]:
                if haystack[i:i+len(needle)] == needle:
                    return i
                else:
                    continue
        return n


def my_code_35(nums, target):
    """搜索插入位置"""
    for i,v in enumerate(nums):
        if v == target:
            return i
    if len(nums) == 1:
        if nums[0] < target:
            return 1
        else:
            return 0
    for i,v in enumerate(nums):
        if i >= len(nums)-1:
            return len(nums)
        else:
            if nums[i] < target < nums[i + 1]:
                return i + 1
            elif target < nums[i]:
                return 0


def my_code_58(s):
    """最后一个单词的长度"""
    n = 0
    for i in range(1, len(s)):
        if s[-i] == ' ':
            continue
        elif s[-i] != ' ' and s[-i-1] == ' ':
            break
        else:
            n += 1
    return n+1


def my_code_66(l):
    """加一"""
    for i in range(1, len(l)+1):
        if l[-i] == 9:
            l[-i] = 0
            if i == len(l) and l[-i] == 0:
                l.insert(0, 1)
                break
        else:
            l[-i] = l[-i] + 1
            break
    return l


def my_code_67(a, b):
    """二进制求和"""
    if len(a) < len(b):
        a = a.zfill(len(b))
    else:
        b = b.zfill(len(a))
    a = list(a)
    b = list(b)
    is_jw = 0
    for i in range(1, len(b)+1):
        c = int(b[-i])+int(a[-i])+is_jw
        if c >= 2:
            a[-i] = str(c % 2)
            if i == len(b):
                a[-i] = str(c % 2)
                a.insert(0, '1')
                break
            is_jw = 1
        else:
            a[-i] = str(c)
            is_jw = 0
    res = ''.join(a)
    return res


if __name__ == '__main__':
    aa = '110010101111'
    bb = '11110111'
    re = my_code_67(aa,bb)
    print(re)
    # l1 = create_linked_list([0])
    # l2 = create_linked_list([0])
    # re = my_code_21(l1, l2)
    # print_linked_list(re)