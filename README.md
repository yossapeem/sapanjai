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

To be able to use the model globally from Replicate:

1. Request global access by emailing **[insert email address here]**.
2. Once you receive your Replicate API token, add it to your `.env` file:

```env
REPLICATE_API_TOKEN=<your_replicate_api_token_here>

### Terminal 1: Frontend Setup
Navigate to the project directory:
```
cd Sapanjai
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
cd Sapanjai
```

Start Supabase locally (Docker needs to be openned) :
```
npx supabase start
```

Serve Supabase functions:
```
npx supabase functions serve
```

2. Detailed User Manual (คู่มือการใช้งานโดยละเอียด)
------------------------------------------------------

Follow every step thoroughly for both terminals as listed above.

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

**Notice** :We may temporarily downgrade the server to save costs, and as a result, the processing might become significantly slower for the first message, but it will still remain functional.



