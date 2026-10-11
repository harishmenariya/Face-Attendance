import sqlite3
import hashlib
from datetime import datetime

DB_PATH = "attendance.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            registered_date TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_code TEXT,
            name TEXT,
            date TEXT,
            time TEXT,
            UNIQUE(user_code, date)
        )
    ''')

    cursor.execute("SELECT * FROM admins WHERE username = 'admin'")
    if not cursor.fetchone():
        hashed_pw = hashlib.sha256("admin123".encode()).hexdigest()
        cursor.execute("INSERT INTO admins (username, password_hash) VALUES (?, ?)", ("admin", hashed_pw))

    conn.commit()
    conn.close()

def verify_admin(username, password) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute("SELECT * FROM admins WHERE username = ? AND password_hash = ?", (username, hashed_pw))
    result = cursor.fetchone() is not None
    conn.close()
    return result

def add_user(user_code, name, reg_date=None) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if not reg_date:
        reg_date = datetime.now().strftime('%Y-%m-%d')
        
    try:
        cursor.execute("INSERT INTO users (user_code, name, registered_date) VALUES (?, ?, ?)", 
                       (user_code, name, reg_date))
        conn.commit()
        numeric_id = cursor.lastrowid
        return numeric_id
    except sqlite3.IntegrityError:
        return -1
    finally:
        conn.close()

def update_user(numeric_id, new_user_code, new_name, new_reg_date=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT user_code FROM users WHERE id = ?", (numeric_id,))
        row = cursor.fetchone()
        if not row:
            return False, "User not found."
        
        old_user_code = row[0]
        
        if new_reg_date:
            cursor.execute("UPDATE users SET user_code = ?, name = ?, registered_date = ? WHERE id = ?",
                           (new_user_code, new_name, new_reg_date, numeric_id))
        else:
            cursor.execute("UPDATE users SET user_code = ?, name = ? WHERE id = ?",
                           (new_user_code, new_name, numeric_id))
                           
        cursor.execute("UPDATE attendance SET user_code = ?, name = ? WHERE user_code = ?",
                       (new_user_code, new_name, old_user_code))
                       
        conn.commit()
        return True, "User updated successfully."
    except sqlite3.IntegrityError:
        return False, "User code already exists."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def delete_user(numeric_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT user_code FROM users WHERE id = ?", (numeric_id,))
        row = cursor.fetchone()
        if not row:
            return False, "User not found."
            
        user_code = row[0]
        
        cursor.execute("DELETE FROM users WHERE id = ?", (numeric_id,))
        cursor.execute("DELETE FROM attendance WHERE user_code = ?", (user_code,))
        
        conn.commit()
        return True, "User deleted successfully."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def get_all_users() -> list:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, user_code, name, registered_date FROM users ORDER BY id ASC")
    results = cursor.fetchall()
    conn.close()
    return results

def get_user_by_id(numeric_id) -> tuple:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT user_code, name FROM users WHERE id = ?", (numeric_id,))
    result = cursor.fetchone()
    conn.close()
    return result

def mark_attendance(user_code, name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    
    try:
        cursor.execute("INSERT INTO attendance (user_code, name, date, time) VALUES (?, ?, ?, ?)",
                       (user_code, name, date_str, time_str))
        conn.commit()
        return True, time_str
    except sqlite3.IntegrityError:
        # Already punched in today
        # Fetch the existing time
        cursor.execute("SELECT time FROM attendance WHERE user_code = ? AND date = ?", (user_code, date_str))
        existing_time = cursor.fetchone()[0]
        return False, existing_time
    finally:
        conn.close()

def get_today_attendance() -> list:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    date_str = datetime.now().strftime('%Y-%m-%d')
    cursor.execute("SELECT user_code, name, date, time FROM attendance WHERE date = ? ORDER BY entry_id DESC", (date_str,))
    results = cursor.fetchall()
    conn.close()
    return results

def get_daily_attendance_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    cursor.execute("SELECT COUNT(id) FROM users")
    total = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(entry_id) FROM attendance WHERE date = ?", (date_str,))
    present = cursor.fetchone()[0] or 0
    
    absent = max(0, total - present)
    percentage = round((present / total * 100), 1) if total > 0 else 0.0
    
    conn.close()
    return {'total': total, 'present': present, 'absent': absent, 'percentage': percentage}

def get_user_attendance_summary() -> list:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(DISTINCT date) FROM attendance")
    total_working_days = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT id, user_code, name FROM users ORDER BY id ASC")
    users = cursor.fetchall()
    
    summary = []
    for u in users:
        uid, user_code, name = u
        cursor.execute("SELECT COUNT(DISTINCT date) FROM attendance WHERE user_code = ?", (user_code,))
        present_days = cursor.fetchone()[0] or 0
        absent_days = max(0, total_working_days - present_days)
        percentage = f"{round((present_days / total_working_days * 100), 1)}%" if total_working_days > 0 else "0.0%"
        
        summary.append({
            'id': uid,
            'user_code': user_code,
            'name': name,
            'total_days': total_working_days,
            'present': present_days,
            'absent': absent_days,
            'percentage': percentage
        })
        
    conn.close()
    return summary

def get_daily_report(date_str=None) -> dict:
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT u.id, u.user_code, u.name, a.time 
        FROM users u
        LEFT JOIN attendance a ON u.user_code = a.user_code AND a.date = ?
        ORDER BY u.id ASC
    """, (date_str,))
    rows = cursor.fetchall()
    conn.close()
    
    records = []
    present_count = 0
    for r in rows:
        is_present = r[3] is not None
        if is_present:
            present_count += 1
        records.append({
            "uid": r[0],
            "user_code": r[1],
            "name": r[2],
            "date": date_str,
            "punch_time": r[3] if is_present else "--",
            "status": "Present" if is_present else "Absent"
        })
        
    total_users = len(records)
    rate = round((present_count / total_users * 100), 1) if total_users > 0 else 0.0
    
    return {
        "date": date_str,
        "total_users": total_users,
        "present_count": present_count,
        "absent_count": total_users - present_count,
        "attendance_rate": rate,
        "records": records
    }

def get_monthly_report(month_str=None) -> dict:
    if not month_str:
        month_str = datetime.now().strftime("%Y-%m")
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(DISTINCT date) 
        FROM attendance 
        WHERE strftime('%Y-%m', date) = ?
    """, (month_str,))
    total_days_conducted = cursor.fetchone()[0] or 1
    
    cursor.execute("""
        SELECT u.id, u.user_code, u.name, 
               COUNT(DISTINCT a.date) as days_present
        FROM users u
        LEFT JOIN attendance a ON u.user_code = a.user_code AND strftime('%Y-%m', a.date) = ?
        GROUP BY u.id
        ORDER BY u.id ASC
    """, (month_str,))
    rows = cursor.fetchall()
    conn.close()
    
    records = []
    for r in rows:
        p_days = r[3]
        rate = round((p_days / total_days_conducted * 100), 1)
        records.append({
            "uid": r[0],
            "user_code": r[1],
            "name": r[2],
            "month": month_str,
            "days_present": p_days,
            "days_absent": max(0, total_days_conducted - p_days),
            "total_days": total_days_conducted,
            "attendance_percentage": rate
        })
        
    return {
        "month": month_str,
        "working_days": total_days_conducted,
        "records": records
    }

def get_user_monthly_report(numeric_id: int, month_str=None) -> dict:
    if not month_str:
        month_str = datetime.now().strftime("%Y-%m")
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get user details
    cursor.execute("SELECT id, user_code, name FROM users WHERE id = ?", (numeric_id,))
    user = cursor.fetchone()
    if not user:
        conn.close()
        return {"status": "error", "message": "User not found"}
        
    # Total unique active days conducted in this month
    cursor.execute("SELECT COUNT(DISTINCT date) FROM attendance WHERE strftime('%Y-%m', date) = ?", (month_str,))
    total_working = cursor.fetchone()[0] or 0
    effective_days = max(1, total_working)
    
    # Day-by-day punches of this specific user in this month
    cursor.execute("""
        SELECT date, time 
        FROM attendance 
        WHERE user_code = ? AND strftime('%Y-%m', date) = ?
        ORDER BY date ASC, time ASC
    """, (user[1], month_str))
    punches = cursor.fetchall()
    conn.close()
    
    # We want unique days present
    days_present = len(set([p[0] for p in punches]))
    attendance_rate = round((days_present / effective_days * 100), 1) if total_working > 0 else 0.0
    
    return {
        "status": "success",
        "uid": user[0],
        "user_code": user[1],
        "name": user[2],
        "month": month_str,
        "total_working_days": total_working,
        "days_present": days_present,
        "days_absent": max(0, total_working - days_present),
        "attendance_rate": attendance_rate,
        "daily_breakdown": [{"date": p[0], "punch_time": p[1], "status": "Present"} for p in punches]
    }


if __name__ == '__main__':
    print("=== STEP 2: db_manager.py SELF-TEST ===")
    
    print("1. Initializing DB...")
    init_db()
    
    print("2. Verifying default admin (admin / admin123)...")
    if verify_admin("admin", "admin123"):
        print("  [OK] Admin verification successful.")
    else:
        print("  [FAIL] Admin verification failed.")
        
    print("3. Testing user creation...")
    uid = add_user("TEST_001", "Test User")
    if uid != -1:
        print(f"  [OK] User created successfully with ID: {uid}.")
    else:
        print("  [INFO] User already exists.")
        
    print("4. Testing attendance marking...")
    success, t = mark_attendance("TEST_001", "Test User")
    if success:
        print(f"  [OK] Attendance marked successfully at {t}.")
    else:
        print(f"  [INFO] Attendance already marked today at {t}.")
        
    print("5. Running stats queries...")
    stats = get_daily_attendance_stats()
    print(f"  [OK] Daily Stats: {stats}")
    
    summary = get_user_attendance_summary()
    print(f"  [OK] User Summary retrieved. Found {len(summary)} user(s).")
    
    print("=== SELF-TEST COMPLETE ===")
