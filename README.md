# 🧠 Sapanjai App

Sapanjai is a sentiment-aware chat application designed to detect and respond to users' emotions in real time using a FastAPI backend and Supabase. Built with React Native and Expo.

---

### Disclaimer (ข้อตกลงในการใช้ซอฟต์แวร์)
------------------------------------------

ซอฟต์แวร์นี้เป็นผลงานที่พัฒนาขึ้นโดย  
นายยศพีร์ มนัสปิยะ  
นางสาววิริญจน์ ชินธรรมมิตร  
นายภัทรพล คงสุข  
จากโรงเรียนสาธิตนานาชาติมหาวิทยาลัยมหิดล  
ภายใต้การดูแลของนายดาวิษ เทพชาตรี  
ภายใต้โครงการ **ระบบปัญญาประดิษฐ์เพื่อการสื่อสารอย่างปลอดภัยสำหรับผู้มีภาวะเปราะบางทางจิต**  
ซึ่งสนับสนุนโดยสำนักงานพัฒนาวิทยาศาสตร์และเทคโนโลยีแห่งชาติ (สวทช.)

โดยมีวัตถุประสงค์เพื่อส่งเสริมให้นักเรียนและนักศึกษาได้เรียนรู้และฝึกทักษะในการพัฒนาซอฟต์แวร์  
ลิขสิทธิ์ของซอฟต์แวร์นี้จึงเป็นของผู้พัฒนา ซึ่งผู้พัฒนาได้อนุญาตให้สำนักงานพัฒนาวิทยาศาสตร์และเทคโนโลยีแห่งชาติเผยแพร่ซอฟต์แวร์นี้  
ตาม “ต้นฉบับ” โดยไม่มีการแก้ไขดัดแปลงใด ๆ ทั้งสิ้น ให้แก่บุคคลทั่วไปได้ใช้เพื่อประโยชน์ส่วนบุคคลหรือเพื่อการศึกษา  
โดยไม่มีวัตถุประสงค์ในเชิงพาณิชย์และไม่คิดค่าตอบแทนจากการใช้งานซอฟต์แวร์ดังกล่าว

สำนักงานพัฒนาวิทยาศาสตร์และเทคโนโลยีแห่งชาติจึงไม่มีหน้าที่ในการดูแล บำรุงรักษา  
จัดการอบรมการใช้งาน หรือพัฒนาประสิทธิภาพซอฟต์แวร์  
รวมทั้งไม่รับรองความถูกต้องหรือประสิทธิภาพการทำงานของซอฟต์แวร์  
ตลอดจนไม่รับประกันความเสียหายต่าง ๆ อันเกิดจากการใช้ซอฟต์แวร์นี้ทั้งสิ้น

---

This software is a work developed by:

Mr. Yossapee Manaspiya  
Ms. Wirin Chinthammit  
Mr. Pattarapol Kongsuk  

from Mahidol University International Demonstration School under the supervision of Mr. Dawit Thepchatree,  
as part of the project “**Safe Communication with Mental Health-Sensitive Individuals through Artificial Intelligence**”  
which has been supported by the National Science and Technology Development Agency (NSTDA).

The purpose of the project is to encourage students to learn and develop skills in software development.  
Therefore, the intellectual property of this software belongs to the developers.  

The developers grant NSTDA permission to distribute this software as-is, without modification,  
for personal or academic use only, and not for commercial purposes.

NSTDA is not responsible for the maintenance, training, performance, or functionality of the software,  
nor will it be liable for any damages arising from its use.


1. Detailed Installation Manual
-------------------------------

Follow this GitHub directory and download all the files attached:
https://github.com/yossapeem/sapanjai.git

### Terminal 1: Frontend Setup
Navigate to the project directory:
```
cd sapanjai
```

Install the necessary frontend dependencies:
```
npm install
```

If you encounter errors during installation, use this command instead:
```
npm install --legacy-peer-deps
```

### Terminal 2: Database Setup
Install Docker (if not already installed):
Follow the installation guide here: https://www.docker.com/get-started/

Install Deno (required for Supabase):
Follow the installation guide here: https://deno.land/manual/getting_started/installation

Navigate to the Sapanjai directory:
```
cd sapanjai
```

Start Supabase locally:
```
npx supabase start
```

Log in to Supabase:
```
npx supabase login
```

(Use your Supabase credentials)
Email: pattarapol.ksk@gmail.com  
Password: Whatisgrass_04  
Enter the verification code shown on screen.

Serve Supabase functions:
```
npx supabase functions serve
```

### Terminal 3: Backend Setup
Install FastAPI, Uvicorn, and other dependencies:
```
pip install fastapi uvicorn transformers torch
```
Optional: Set up a virtual environment:
```
python -m venv venv
```

MacOS/Linux: source venv/bin/activate  
Windows: venv\Scripts\activate

If you have a `requirements.txt`, install dependencies with:
```
pip install -r requirements.txt
```

2. Detailed User Manual (คู่มือการใช้งานโดยละเอียด)
------------------------------------------------------

Follow every step thoroughly for all three terminals as listed above.

Start the FastAPI server with this command in Terminal 3:
```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

If you get an error saying `uvicorn` is not recognized, use:
```
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

To access the platform, install **Expo Go** from the App Store or Google Play Store.

Start the application with this command in Terminal 1:
```
npx expo start
```

After running that command, a QR code should appear.

Use your mobile device to scan the QR code:
- iOS: Use the Camera app
- Android: Use the in-app camera

**Notice**: On the registration page, you do not need to sign in with a legitimate email account.


