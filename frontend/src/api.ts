export const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export type User={id:string;name:string;email:string;role:string;organization:{id:string;name:string}};
export type Job={id:string;status:'PENDING'|'RUNNING'|'COMPLETED'|'FAILED';question:string;error_message?:string;report_id?:string;created_at:string;started_at?:string;completed_at?:string};
export type Report={id:string;title:string;executive_summary:string;report_data:any;created_at:string};

class Api {
  token=localStorage.getItem('equitylens_token');
  async request<T>(path:string, init:RequestInit={}) {
    const url=API_BASE+path, method=init.method || 'GET';
    if(path==='/research' && method==='POST') { console.info('[RESEARCH_UI] request URL:',url); console.info('[RESEARCH_UI] request started',{method,authenticated:Boolean(this.token)}); }
    try {
      const response=await fetch(url,{...init,headers:{'Content-Type':'application/json',...(this.token?{Authorization:`Bearer ${this.token}`}:{...{}}),...init.headers}});
      if(path==='/research' && method==='POST') console.info('[RESEARCH_UI] response status:',response.status);
      const body=await response.json();
      if(!response.ok) throw new Error(body.error?.message||'Request failed');
      return body.data as T;
    } catch(error) {
      if(path==='/research' && method==='POST') console.error('[RESEARCH_UI] request failed:',error);
      throw error;
    }
  }
  async auth(path:string, body:unknown){const data=await this.request<{access_token:string;user:User}>(path,{method:'POST',body:JSON.stringify(body)});this.token=data.access_token;localStorage.setItem('equitylens_token',data.access_token);return data.user}
  login=(email:string,password:string)=>this.auth('/auth/login',{email,password}); signup=(value:any)=>this.auth('/auth/signup',value); me=()=>this.request<User>('/auth/me'); jobs=()=>this.request<{items:Job[]}>('/research'); clearResearchHistory=()=>this.request<{deleted_jobs:number}>('/research',{method:'DELETE'});
  create=(ticker:string,question:string)=>{console.info('[RESEARCH_UI] calling createResearch');return this.request<Job>('/research',{method:'POST',body:JSON.stringify({ticker,question})})};
  job=(id:string)=>this.request<Job>(`/research/${id}`); report=(id:string)=>this.request<Report>(`/reports/${id}`); logout(){this.token=null;localStorage.removeItem('equitylens_token');}
}

export const api=new Api();
