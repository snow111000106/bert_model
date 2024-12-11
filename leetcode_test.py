class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def create_linked_list(vals):
    dummy = ListNode()
    current = dummy
    for val in vals:
        current.next = ListNode(val)
        current = current.next
    return dummy.next


def print_linked_list(head):
    current = head
    vals = []
    while current:  # 遍历链表直到末尾
        vals.append(current.val)  # 将当前节点的值加入列表
        current = current.next       # 移动到下一个节点
    print(" -> ".join(map(str, vals)))


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
        elif current_l1.val <= current_l2.val:
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


def my_code_69(x):
    """x的平方根"""
    n = 0
    while n*n <= x:
        n += 1
    return n-1


def my_code_69_2(x):
    """x的平方根-解法2"""
    l = 0
    r = x
    a = 0
    while l <= r:
        mid = (l+r)//2
        if mid * mid <= x:
            a = mid
            l = mid + 1
        else:
            r = mid - 1
    return a


def math_a(n):
    p = 1
    for i in range(1, n+1):
        p *= i
    return p


def math_comb(m, n):
    ans = math_a(m) // (math_a(m-n)*math_a(n))
    return ans


def my_code_70(x):
    """爬楼梯-组合解法"""
    i = 1
    count = 1
    while i <= x:
        if 2*i > x:
            break
        elif 2*i == x:
            count = count + 1
            break
        else:
            count = count+math_comb(x-i, i)
            i += 1
    return count


def my_code_70_2(n):
    """爬楼梯-斐波那契数列"""
    p = 0
    q = 1
    s = 1
    for i in range(1, n+1):
        s = p+q
        p = q
        q = s
    return s


def my_code_83(head):
    """删除排序链表中的重复值"""
    if not head:  # 如果链表为空，直接返回
        return None
    current = head
    while current.next:
        if current.val == current.next.val:
            current.next = current.next.next
        else:
            current = current.next

    return head


def my_code_88(nums1, m, nums2, n):
    """合并两个有序数组"""
    for i in range(n):
        nums1.insert(m+i, nums2[i])
        nums1.pop()
    nums1.sort()
    print(f'nums1:{nums1}')


def my_code_88_2(nums1, m, nums2, n):
    """合并两个有序数组-解法2"""
    p1 = m - 1  # nums1 中有效元素的最后一个索引
    p2 = n - 1  # nums2 中的最后一个索引
    p = m + n - 1  # nums1 最后位置的索引（合并后的最终长度）

    # 从后往前遍历，将较大的元素放在 nums1 的最后
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1

    # 如果 nums2 中还有剩余元素，将其填入 nums1
    while p2 >= 0:
        nums1[p] = nums2[p2]
        p2 -= 1
        p -= 1

    print(f'nums1:{nums1}')


def my_code_94(root):
    """二叉树的中序遍历"""
    if not root:
        return []
    rest = []
    rest.extend(my_code_94(root.left))
    rest.append(root.val)
    rest.extend(my_code_94(root.right))
    return rest


if __name__ == '__main__':
    # aa = [-1,0,2,3,0,0,0]
    # bb = [-1,2,3]
    # mm, nn = 4, 3
    root = TreeNode(1)  # 树的根节点
    root.left = TreeNode(2)  # 根的左子树
    root.right = TreeNode(3)  # 根的右子树
    root.left.left = TreeNode(4)  # 左子树的左子树
    root.left.right = TreeNode(5)  # 左子树的右子树
    root.right.left = TreeNode(6)  # 右子树的左子树
    root.right.right = TreeNode(7)  # 右子树的右子树
    r = my_code_94(root)
    print(r)
    # print_linked_list(re)
    l1 = create_linked_list([0])
    # l2 = create_linked_list([0])
    # re = my_code_21(l1, l2)
    # print_linked_list(re)
