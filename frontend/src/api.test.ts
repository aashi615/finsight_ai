import {beforeEach, expect, it, vi} from 'vitest';
import {api} from './api';
beforeEach(()=>{localStorage.clear(); api.token=null;});
it('stores a token after login', async()=>{vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>({data:{access_token:'token',user:{id:'1',name:'A',email:'a@b.com',role:'ADMIN',organization:{id:'o',name:'Org'}}}})})); await api.login('a@b.com','password'); expect(localStorage.getItem('equitylens_token')).toBe('token');});
